#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
终极版：支持对全新数据预测，完全对齐训练流程：
1. 图锚点 = outer_train = inner_train + inner_val（来自 split.json）
2. 新数据格式：第1列=label，第2列起=features（兼容无category）
3. 自动计算 R²（如果新数据提供 label）
4. 特征名自动对齐（缺失补0）
"""
import sys
_ORIG_ARGV = sys.argv[:]
sys.argv = [sys.argv[0]]

import argparse, json, os
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler




class LogTargetScaler:
    """与训练端保持一致的目标变换器（仅推理需要 inverse_transform 即可）。
    - 假定训练端保存了该类实例到 y_scaler.pth；此处提供同名定义便于反序列化。
    - 若训练用的是 StandardScaler，这个类不会被用到。
    """
    def __init__(self, base: float = 10.0, eps: float = None, with_std: bool = False):
        self.base = float(base)
        self.eps = eps
        self.with_std = bool(with_std)
        self._std = None
        self._mean_ = 0.0
        self._scale_ = 1.0
        self.eps_ = None
        # 训练端可能存的属性（做数值保护）
        self.z_min_ = None
        self.z_max_ = None
        self.y_max_ = None
        self.clip_margin = 1.0

    # 推理阶段只需要 inverse_transform；transform/fit 可不严格实现
    def inverse_transform(self, z):
        import numpy as np
        z = np.asarray(z).reshape(-1)
        # 去标准化
        if self._std is not None:
            z = z * (getattr(self, "_scale_", 1.0) + 1e-12) + getattr(self, "_mean_", 0.0)
        # 裁剪 z，避免 10**z 溢出
        z_lo = (self.z_min_ if self.z_min_ is not None else -50.0) - self.clip_margin
        z_hi = (self.z_max_ if self.z_max_ is not None else  50.0) + self.clip_margin
        z = np.clip(z, z_lo, z_hi)
        # 反变换
        eps = self.eps_ if getattr(self, "eps_", None) is not None else (self.eps if self.eps is not None else 0.0)
        y = (self.base ** z) - eps
        # 兜底清理
        upper = (self.y_max_ * 1e3) if (self.y_max_ is not None) else 1e12
        y = np.nan_to_num(y, nan=0.0, posinf=upper, neginf=0.0)
        return y


# === 工具函数（与训练一致）===
def to_tensor(x):
    return torch.tensor(x, dtype=torch.float32)

def inv_if_scaler(arr, scaler):
    if scaler is None:
        return arr
    return scaler.inverse_transform(arr.reshape(-1, 1)).ravel()

def load_sheet_for_new_data(xlsx_path, sheet_name):
    """
    专为新数据设计：
    - 第1列：label（必须）
    - 第2列起：features
    - 如果有第0列，当作 category；否则 category = "Unknown"
    """
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name)
    if df.shape[1] < 2:
        raise ValueError(f"新数据 '{sheet_name}' 至少需要两列：label, features...")

    y = df.iloc[:, 0].values.astype(np.float32)
    X = df.iloc[:, 1:].values.astype(np.float32)
    feat_names = [str(c) for c in df.columns[1:]]

    row_ids = np.arange(len(df), dtype=np.int64)
    return X, y, feat_names, row_ids

def load_sheet_for_anchor(xlsx_path, sheet_name):
    """加载原始训练数据（用于构建图锚点）"""
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name)
    if df.shape[1] < 3:
        raise ValueError(f"训练数据 '{sheet_name}' 至少需要三列：category, label, features...")
    cats = df.iloc[:, 0].values
    y = df.iloc[:, 1].values.astype(np.float32)
    X = df.iloc[:, 2:].values.astype(np.float32)
    feat_names = [str(c) for c in df.columns[2:]]
    row_ids = np.arange(len(df), dtype=np.int64)
    return X, y, cats, feat_names, row_ids

# ====== 导入模型类 ======
from model.embedding_net import EmbeddingMLP
from model.transformerModel import SimpleTransformerRegressor
from descriptors_group.getGCNDescriptors import (
    knn_train, l1_norm, norm_adj_train, knn_val, norm_adj_val
)

# === 嵌入函数 ===
def embed(model, X, device):
    x_s = model._x_scaler.transform(X)
    with torch.no_grad():
        e = model(to_tensor(x_s).to(device)).cpu().numpy()
    return e

# === 为新数据构建 GCN 特征（锚点 = outer_train）===
def build_gcn_features_for_new_data(X_anchor, X_new, feat_names, model, device, k=10, a=0.1,
                                    grouped=True, mapping=None):
    """
    X_anchor: outer_train 数据（inner_train + inner_val）
    X_new: 待预测的新数据
    """
    E_anchor = embed(model, X_anchor, device)
    E_new = embed(model, X_new, device)

    S_new = np.dot(E_new, E_anchor.T)  # 新数据只连训练锚点

    # 构建锚点图（固定）
    S_anchor = np.dot(E_anchor, E_anchor.T)
    tr_knn = knn_train(S_anchor, k)
    tr_norm = l1_norm(tr_knn)
    A_anchor, deg_anchor = norm_adj_train(tr_norm, a)

    X_anchor_s = model._x_scaler.transform(X_anchor).astype(np.float32)
    X_new_s = model._x_scaler.transform(X_new).astype(np.float32)

    n_anchor = X_anchor_s.shape[0]
    n_new = X_new_s.shape[0]

    # 融合锚点自身（保持一致性）
    X_anchor_fused = np.dot(A_anchor.astype(np.float32), X_anchor_s)

    # 构建新数据连接到锚点的图
    tr_block_new = np.hstack([A_anchor.astype(np.float32), np.zeros((n_anchor, n_new), dtype=np.float32)])
    new_block = knn_val(S_new, n_new, n_anchor + n_new, k).astype(np.float32)
    full_A_new = np.vstack([tr_block_new, new_block])
    A_new = norm_adj_val(full_A_new, deg_anchor, a)
    X_stack_new = np.vstack([X_anchor_s, X_new_s]).astype(np.float32)
    Xnew_fused = np.dot(A_new, X_stack_new)[-n_new:]  # 只取新数据部分

    # 分组或不分组
    if grouped:
        if mapping is None:
            raise ValueError("Grouped=True 需要 descriptors mapping JSON。")
        max_cols = max(len(v) for v in mapping.values())
        group_order = list(mapping.keys())
        name2idx = {nm: i for i, nm in enumerate(feat_names)}

        def to_seq(mat):
            rows = []
            for r in range(mat.shape[0]):
                seq = []
                for g in group_order:
                    vals = [float(mat[r, name2idx[c]]) if c in name2idx else 0.0 for c in mapping.get(g, [])]
                    while len(vals) < max_cols:
                        vals.append(0.0)
                    seq.append(vals)
                rows.append(seq)
            return np.array(rows, dtype=np.float32)

        Xnew_seq = to_seq(Xnew_fused)
        seq_len = len(group_order)
        input_dim = max_cols
        return Xnew_seq, seq_len, input_dim, group_order
    else:
        Xnew_seq = Xnew_fused[:, None, :].astype(np.float32)
        return Xnew_seq, 1, Xnew_fused.shape[1], feat_names

# === 安全加载 ===
def safe_load(path, device):
    try:
        from torch.serialization import safe_globals
        with safe_globals([EmbeddingMLP, SimpleTransformerRegressor, StandardScaler, LogTargetScaler]):
            return torch.load(path, map_location=device)
    except Exception:
        return torch.load(path, map_location=device, weights_only=False)

def main():
    sys.argv = _ORIG_ARGV

    ap = argparse.ArgumentParser()
    ap.add_argument('--fold_dir', required=True, help='包含 embedding_best.pth / transformer_best.pth / y_scaler.pth / split.json 的目录')
    ap.add_argument('--xlsx_train', required=True, help='原始训练数据 all_descriptors.xlsx（用于加载 outer_train 锚点）')
    ap.add_argument('--sheet_train', default='Sheet1', help='训练数据的 sheet 名称')
    ap.add_argument('--xlsx_new', required=True, help='新数据 Excel 路径（格式：label, features... 或 category, label, features...）')
    ap.add_argument('--sheet_new', default='Sheet1', help='新数据的 sheet 名称')
    ap.add_argument('--mapping', default=None, help='(可选) descriptors 分组 JSON；不填则使用 split.json 中的 mapping_path')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--out_csv', default='new_data_predictions.csv')
    args = ap.parse_args()

    device = torch.device(args.device)

    # 1) 读取 split.json 中的索引和超参
    split_path = os.path.join(args.fold_dir, 'split.json')
    if not os.path.exists(split_path):
        raise FileNotFoundError("缺少 split.json，无法知道训练时的划分与参数: " + split_path)
    with open(split_path, 'r', encoding='utf-8') as f:
        split = json.load(f)

    k = int(split.get('k', 10))
    a = float(split.get('a', 0.1))
    grouped = bool(split.get('grouped', True))
    mapping_path = args.mapping or split.get('mapping_path', None)
    if grouped and (mapping_path is None or not os.path.exists(mapping_path)):
        raise FileNotFoundError("需要 descriptors 分组 JSON，请通过 --mapping 指定，或确保 split.json 的 mapping_path 可用。")

    # 关键修改：使用 outer_train = inner_train + inner_val 作为锚点
    inner_train_idx = [int(x) for x in split['inner_train_idx']]
    inner_val_idx = [int(x) for x in split['inner_val_idx']]
    outer_train_idx = inner_train_idx + inner_val_idx  # 合并！
    print(f"Using outer_train (inner_train + inner_val) as graph anchor, total {len(outer_train_idx)} samples.")

    # 2) 加载 descriptors mapping（如果 grouped）
    descriptors_mapping = None
    if grouped:
        with open(mapping_path, 'r', encoding='utf-8') as f:
            descriptors_mapping = json.load(f)

    # 3) 加载原始训练数据（用于构建图锚点）
    X_train_all, y_train_all, cats_train_all, feat_names_train, row_ids_train = load_sheet_for_anchor(args.xlsx_train, args.sheet_train)
    idx2pos_train = {int(rid): pos for pos, rid in enumerate(row_ids_train.tolist())}
    outer_pos = np.array([idx2pos_train[i] for i in outer_train_idx], dtype=int)
    X_anchor = X_train_all[outer_pos]  # 图锚点 = outer_train
    print(f"Loaded {len(X_anchor)} samples as graph anchor from outer_train (inner_train + inner_val).")

    # 4) 加载新数据（第1列=label，第2列起=features）
    X_new, y_new, feat_names_new, row_ids_new = load_sheet_for_new_data(args.xlsx_new, args.sheet_new)
    print(f"Loaded {len(X_new)} new samples for prediction.")

    # 5) 特征对齐（新数据特征名 vs 训练数据特征名）
    if feat_names_train != feat_names_new:
        print("[Warning] 新数据特征名与训练数据不一致，将按名称对齐（缺失补0）")
        name2idx_train = {nm: i for i, nm in enumerate(feat_names_train)}
        X_new_aligned = np.zeros((X_new.shape[0], len(feat_names_train)), dtype=np.float32)
        for i, nm in enumerate(feat_names_new):
            if nm in name2idx_train:
                X_new_aligned[:, name2idx_train[nm]] = X_new[:, i]
        X_new = X_new_aligned
        feat_names_new = feat_names_train

    # 6) 加载模型与 scaler
    embedding_path = os.path.join(args.fold_dir, 'embedding_best.pth')
    transformer_path = os.path.join(args.fold_dir, 'transformer_best.pth')
    y_scaler_path = os.path.join(args.fold_dir, 'y_scaler.pth')

    if not os.path.exists(embedding_path):   raise FileNotFoundError(embedding_path)
    if not os.path.exists(transformer_path): raise FileNotFoundError(transformer_path)
    if not os.path.exists(y_scaler_path):    raise FileNotFoundError(y_scaler_path)

    embed_model = safe_load(embedding_path, device)
    trans_model = safe_load(transformer_path, device)
    y_scaler = safe_load(y_scaler_path, device)

    embed_model.eval()
    trans_model.eval()

    # 7) 构建新数据的 GCN 融合特征（连接到 outer_train 锚点）
    Xnew_seq, seq_len, input_dim, group_order = build_gcn_features_for_new_data(
        X_anchor, X_new, feat_names_train, embed_model, device,
        k=k, a=a, grouped=grouped, mapping=descriptors_mapping
    )

    # 8) 预测 + 反归一化
    with torch.no_grad():
        y_pred_norm = trans_model(to_tensor(Xnew_seq).to(device)).cpu().numpy().ravel()
        y_pred = inv_if_scaler(y_pred_norm, y_scaler)

    print(f"[Prediction Done] seq_len={seq_len}, input_dim={input_dim}, samples={len(y_pred)}")

    out_data = {
		"y_true": y_new.astype(float),
		"y_pred": y_pred.astype(float),
	}

    valid_mask = (
		pd.notna(y_new) & pd.notna(y_pred) &
		np.isfinite(y_new) & np.isfinite(y_pred)
	)

    if valid_mask.sum() >= 2:
        r2 = float(r2_score(y_new[valid_mask], y_pred[valid_mask]))
        print(f"[Evaluation] R² on new data = {r2:.6f}")
        out_data["R2"] = [r2] + [np.nan] * (len(y_pred) - 1)
    elif valid_mask.sum() == 1:
        print("[Warn] Only 1 valid sample, R² is undefined (set to NaN)")
        out_data["R2"] = [float('nan')] + [np.nan] * (len(y_pred) - 1)
    else:
        print("[Warn] No valid samples for R² calculation")
        out_data["R2"] = [float('nan')] + [np.nan] * (len(y_pred) - 1)

    out = pd.DataFrame(out_data)
    out_path = os.path.join(args.fold_dir, args.out_csv)
    out.to_csv(out_path, index=False)
    print("Saved predictions to:", out_path)

if __name__ == '__main__':
    main()
