#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
复现 + 预测（Option A：预测阶段仅连接到 *inner_train* 锚点）
-----------------------------------------------------------------
- reproduce：直接读取训练阶段保存的 stage1.5 融合特征 + split.json + 模型权重，
             复现 train/val/test 的 R²（**不重建 1.5**）。
- predict：  对新的 all_data.xlsx 风格文件（每个 sheet 一个属性，
             **第一列是 label，第二列起是 Mordred 原始特征**）做预测；
             1.5 阶段的融合特征 **只把新样本连到 inner_train（训练集）**，
             与当前训练端的 1.5 架构保持一致，避免分布错位与数据泄漏。

所需产物（fold_dir 下）：
- embedding_best.pth
- transformer_best.pth
- y_scaler.pth
- split.json  （含 outer/inner 索引、k、a、grouped、mapping_path 等）
- （仅 reproduce 模式需要）stage15_fused_grouped.xlsx（train/val/test 三个 sheet）

依赖你的工程模块（确保可 import）：
- model.embedding_net.EmbeddingMLP
- model.transformerModel.SimpleTransformerRegressor
- descriptors_group.getGCNDescriptors.{knn_train, l1_norm, norm_adj_train, knn_val, norm_adj_val}

Author: your teammate :)
"""
import sys
_ORIG_ARGV = sys.argv[:]
sys.argv = [sys.argv[0]]

import argparse
import json
import os
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

# ====== 兼容训练端的 LogTargetScaler（用于反变换/反对数） ======
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

# ====== allowlist 导入（让 torch.load 能反序列化模型） ======
from model.embedding_net import EmbeddingMLP   # noqa: F401
from model.transformerModel import SimpleTransformerRegressor  # noqa: F401
from descriptors_group.getGCNDescriptors import (
    knn_train, l1_norm, norm_adj_train, knn_val, norm_adj_val
)

# ====== 工具函数 ======

def to_tensor(x):
    return torch.tensor(x, dtype=torch.float32)


def inv_if_scaler(arr, scaler):
    """统一反变换：兼容 StandardScaler / LogTargetScaler；并做数值兜底"""
    if scaler is None:
        return arr
    import numpy as _np
    out = scaler.inverse_transform(arr.reshape(-1, 1)).ravel()
    out = _np.nan_to_num(out, nan=0.0,
                         posinf=(_np.max(out[_np.isfinite(out)]) if _np.any(_np.isfinite(out)) else 1e12),
                         neginf=0.0)
    return out


def safe_load(path, device):
    """更稳妥的反序列化（优先尝试 weights_only=False；必要时允许自定义类）"""
    try:
        # PyTorch 2.6+ 提供 safe_globals；此处做兼容
        from torch.serialization import safe_globals
        with safe_globals([EmbeddingMLP, SimpleTransformerRegressor, StandardScaler, LogTargetScaler]):
            return torch.load(path, map_location=device)
    except Exception:
        return torch.load(path, map_location=device, weights_only=False)


# ====== 读取 sheet（兼容“训练文件的老格式”和“预测文件的新格式”） ======
# 训练文件（old）：第一列可能是 category（字符串），第二列 label，后续是特征
# 预测文件（new）：第一列 label，第二列起是特征（无 category）

def load_sheet_flexible(xlsx_path, sheet_name):
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name)
    if df.shape[1] < 2:
        raise ValueError(f"工作表 '{sheet_name}' 至少需要两列：label, features...")
    # 猜测第一列是否 label：数值比例很高则视为 label-first
    first_is_numeric = pd.api.types.is_numeric_dtype(df.iloc[:, 0])
    second_is_numeric = pd.api.types.is_numeric_dtype(df.iloc[:, 1]) if df.shape[1] > 1 else False
    if first_is_numeric and (df.shape[1] == 2 or second_is_numeric):
        # label-first: [label, f1, f2, ...]
        cats = np.array(["NA"] * len(df))
        y = df.iloc[:, 0].values.astype(np.float32)
        X = df.iloc[:, 1:].values.astype(np.float32)
        feat_names = [str(c) for c in df.columns[1:]]
    else:
        # category-first: [category, label, f1, f2, ...]
        cats = df.iloc[:, 0].values
        y = df.iloc[:, 1].values.astype(np.float32)
        X = df.iloc[:, 2:].values.astype(np.float32)
        feat_names = [str(c) for c in df.columns[2:]]
    row_ids = np.arange(len(df), dtype=np.int64)
    return X, y, cats, feat_names, row_ids


def load_stage15_sheet(xlsx_path, sheet_name):
    """解析训练保存的 stage1.5 融合特征 xlsx（列为 JSON 列表），返回 [N, seq_len, input_dim]"""
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name)
    if df.shape[1] < 3:
        raise ValueError("stage1.5 sheet 至少需要三列：category, label, 以及若干组列")
    cats = df.iloc[:, 0].values
    y    = df.iloc[:, 1].values.astype(np.float32)
    group_cols = list(df.columns[2:])
    N = df.shape[0]
    seq_len = len(group_cols)
    import json as _json
    first_list = _json.loads(str(df.iloc[0, 2]))
    input_dim = len(first_list)
    X = np.zeros((N, seq_len, input_dim), dtype=np.float32)
    for i in range(N):
        for j in range(seq_len):
            lst = _json.loads(str(df.iloc[i, 2 + j]))
            if len(lst) < input_dim:
                lst = lst + [0.0] * (input_dim - len(lst))
            X[i, j, :] = np.array(lst, dtype=np.float32)[:input_dim]
    return X, y, cats, group_cols, input_dim, seq_len, df


# ====== 阶段 1 / 1.5：与训练端保持一致 ======

def embed(model, X, device):
    x_s = model._x_scaler.transform(X)
    with torch.no_grad():
        e = model(to_tensor(x_s).to(device)).cpu().numpy()
    return e


def build_gcn_features_predict_only(X_tr, X_pred, feat_names, model, device, k=10, a=0.1,
                                    grouped=True, mapping=None):
    """仅为“新预测集”构造 Xpred_seq（**Option A：只连 inner_train**）：
       - 训练图 A_tr 仅由 X_tr 构建；
       - 预测样本只与“训练样本”连边（不互连），并用训练度数归一化；
       - 不会把预测样本的信息回流到训练锚点，避免数据泄漏。
    """
    # 1) 嵌入
    E_tr   = embed(model, X_tr,   device)
    E_pred = embed(model, X_pred, device)

    # 2) 相似度（训练-训练，预测-训练）
    S_tr   = np.dot(E_tr,   E_tr.T)
    S_pred = np.dot(E_pred, E_tr.T)

    # 3) 训练图（仅基于训练集）
    tr_knn  = knn_train(S_tr, k)
    tr_norm = l1_norm(tr_knn)
    A_tr, deg_tr = norm_adj_train(tr_norm, a)

    # 4) 标准化特征（沿用训练端的 x_scaler）
    X_tr_s   = model._x_scaler.transform(X_tr).astype(np.float32)
    X_pred_s = model._x_scaler.transform(X_pred).astype(np.float32)

    # 5) 拼接矩阵，仅让预测样本连向训练，不互连
    n_tr, n_pred = X_tr_s.shape[0], X_pred_s.shape[0]
    tr_block_pred = np.hstack([A_tr.astype(np.float32), np.zeros((n_tr, n_pred), dtype=np.float32)])
    pred_block    = knn_val(S_pred, n_pred, n_tr + n_pred, k).astype(np.float32)
    full_A_pred   = np.vstack([tr_block_pred, pred_block])
    A_pred        = norm_adj_val(full_A_pred, deg_tr, a)  # 用训练度数做归一化

    X_stack_pred  = np.vstack([X_tr_s, X_pred_s]).astype(np.float32)
    Xpred_fused   = np.dot(A_pred, X_stack_pred)[-n_pred:]

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

        Xpred_seq = to_seq(Xpred_fused)
        seq_len   = len(group_order)
        input_dim = max_cols
        return Xpred_seq, seq_len, input_dim, group_order
    else:
        Xpred_seq = Xpred_fused[:, None, :].astype(np.float32)
        return Xpred_seq, 1, Xpred_fused.shape[1], feat_names


# =================== 主流程 ===================

def main():
    sys.argv = _ORIG_ARGV

    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', required=True, choices=['reproduce', 'predict'],
                    help='reproduce：用保存的 stage1.5 特征复现；predict：对新数据做预测（仅连 inner_train）')
    ap.add_argument('--fold_dir', required=True,
                    help='训练产物目录（含 embedding_best.pth / transformer_best.pth / y_scaler.pth / split.json）')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')

    # reproduce 专用
    ap.add_argument('--repro_which', default='test', choices=['train','val','test'],
                    help='复现哪个切片的评估（sheet 名称）')

    # predict 专用（Option A：仅连 inner_train 锚点）
    ap.add_argument('--train_xlsx', help='训练时的 all_data.xlsx（原始特征，用于 inner_train 锚点构建）')
    ap.add_argument('--train_sheet', help='训练时的 sheet 名称（用于定位 inner_train 行）')
    ap.add_argument('--pred_xlsx',  help='用于预测的新文件（第一列 label，第二列起原始特征）')
    ap.add_argument('--pred_sheet', help='用于预测的新文件中的 sheet 名称')

    # 输出
    ap.add_argument('--out_csv', default='predictions.csv')

    args = ap.parse_args()
    device = torch.device(args.device)

    # 读取 split.json
    split_path = os.path.join(args.fold_dir, 'split.json')
    if not os.path.exists(split_path):
        raise FileNotFoundError("缺少 split.json: " + split_path)
    with open(split_path, 'r', encoding='utf-8') as f:
        split = json.load(f)

    k        = int(split.get('k', 10))
    a        = float(split.get('a', 0.1))
    grouped  = bool(split.get('grouped', True))
    mapping_path = split.get('mapping_path', None)

    # descriptors 分组
    descriptors_mapping = None
    if grouped:
        if mapping_path is None or not os.path.exists(mapping_path):
            raise FileNotFoundError("需要 descriptors 分组 JSON（mapping_path）。")
        with open(mapping_path, 'r', encoding='utf-8') as f:
            descriptors_mapping = json.load(f)

    # 加载模型与 scaler
    embedding_path   = os.path.join(args.fold_dir, 'embedding_best.pth')
    transformer_path = os.path.join(args.fold_dir, 'transformer_best.pth')
    y_scaler_path    = os.path.join(args.fold_dir, 'y_scaler.pth')
    if not os.path.exists(embedding_path):   raise FileNotFoundError(embedding_path)
    if not os.path.exists(transformer_path): raise FileNotFoundError(transformer_path)
    if not os.path.exists(y_scaler_path):    raise FileNotFoundError(y_scaler_path)

    embed_model = safe_load(embedding_path, device);   embed_model.eval()
    trans_model = safe_load(transformer_path, device); trans_model.eval()
    y_scaler    = safe_load(y_scaler_path, device)

    if args.mode == 'reproduce':
        # 直接读取 stage1.5 特征
        st15_path = os.path.join(args.fold_dir, 'stage15_fused_grouped.xlsx')
        if not os.path.exists(st15_path):
            raise FileNotFoundError("找不到阶段1.5特征文件: " + st15_path)
        X_seq, y, cats, group_cols, input_dim, seq_len, raw_df = load_stage15_sheet(st15_path, args.repro_which)

        with torch.no_grad():
            y_pred_norm = trans_model(to_tensor(X_seq).to(device)).cpu().numpy().ravel()
            y_pred = inv_if_scaler(y_pred_norm, y_scaler)

        r2 = float(r2_score(y, y_pred))
        print(f"[{args.repro_which}] Reproduce R2 = {r2:.6f} (seq_len={seq_len}, input_dim={input_dim})")

        out = pd.DataFrame({
            "category": cats,
            "y_true": y.astype(float),
            "y_pred": y_pred.astype(float),
        })
        out_path = os.path.join(args.fold_dir, args.out_csv)
        out.to_csv(out_path, index=False)
        print("Saved:", out_path)

    else:  # predict（仅连 inner_train 锚点）
        if not args.train_xlsx or not args.train_sheet:
            raise ValueError("--train_xlsx 与 --train_sheet 在 predict 模式下必填（用于 inner_train 锚点）")
        if not args.pred_xlsx or not args.pred_sheet:
            raise ValueError("--pred_xlsx 与 --pred_sheet 在 predict 模式下必填")

        # 读训练数据（用于锚点：strict 使用 inner_train_idx）
        X_all, y_all, cats_all, feat_names, row_ids = load_sheet_flexible(args.train_xlsx, args.train_sheet)
        idx2pos = {int(rid): pos for pos, rid in enumerate(row_ids.tolist())}
        inner_train_idx = [int(x) for x in split['inner_train_idx']]
        tr_pos  = np.array([idx2pos[i] for i in inner_train_idx], dtype=int)
        X_tr, y_tr, c_tr = X_all[tr_pos], y_all[tr_pos], cats_all[tr_pos]

        # 读新数据（label-first 风格）
        X_pred, y_pred_true, cats_pred, feat_names_pred, row_ids_pred = load_sheet_flexible(args.pred_xlsx, args.pred_sheet)

        # 阶段1.5：仅把新样本连向 inner_train（Option A）
        Xpred_seq, seq_len, input_dim, group_order = build_gcn_features_predict_only(
            X_tr, X_pred, feat_names, embed_model, device,
            k=k, a=a, grouped=grouped, mapping=descriptors_mapping
        )

        # 预测 + 反变换
        with torch.no_grad():
            y_hat_norm = trans_model(to_tensor(Xpred_seq).to(device)).cpu().numpy().ravel()
            y_hat = inv_if_scaler(y_hat_norm, y_scaler)

        # out = pd.DataFrame({
        #     "row_id": row_ids_pred.astype(int),
        #     "category": cats_pred,  # 预测文件若无 category，这里为 "NA"
        #     "y_true": y_pred_true.astype(float),
        #     "y_pred": y_hat.astype(float),
        # })
        # out_path = os.path.join(args.fold_dir, args.out_csv)
        # out.to_csv(out_path, index=False)
        # print("Saved:", out_path)

        # 如果新文件包含 label，可打印参考 R²（仅作参考，不等价于严格 test）
        try:
            r2_ref = float(r2_score(y_pred_true, y_hat))
            print(f"[predict] Reference R2 on provided labels = {r2_ref:.6f}")
        except Exception as e:
            print(f"[predict] 无法计算参考 R²：{e}")


if __name__ == '__main__':
    main()
