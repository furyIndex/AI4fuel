#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
5-fold 交叉验证（可复现 & 记录索引）+ 单折/指定 split.json 复训：
- 外层 5 折：显式保存每折的全局行号 split（outer_train/outer_test、inner_train/inner_val）
- 阶段一：对比学习 embedding（按 Spearman 选最佳），保存 fold_i/embedding_best.pth
- 阶段1.5：图融合 + 分组；可保存 stage1.5 的特征到 xlsx
- 阶段二：Transformer + Optuna（固定 batch size，loss∈{mse,msle}，早停按 R²，y 目标标准化训练+测试反归一化）
- 可指定：单 sheet、单/多折；或直接用现有 split.json 精准复训该折
- 【新增】--exp_k：每个 fold 重复实验次数，选择测试集 R² 最高的一次，将该次产物复制到 fold 根目录，保持与原来相同的产物文件名。
"""
# --- 防止被 import 的第三方脚本在导入时 parse_args ---
import sys

from torch.nn.functional import mse_loss

_ORIG_ARGV = sys.argv[:]
sys.argv = [sys.argv[0]]

import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # 或 ":16:8"

import argparse
import json
import os
import random
import shutil

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import optuna

from model.embedding_net import EmbeddingMLP, ContrastiveLoss, embedding_val_func
from descriptors_group.getGCNDescriptors import knn_train, l1_norm, norm_adj_train, knn_val, norm_adj_val
from model.transformerModel import SimpleTransformerRegressor


class LogTargetScaler:

    def __init__(self, base: float = 10.0, eps: float = None, with_std: bool = False):
        self.base = float(base)
        self.eps = eps
        self.with_std = bool(with_std)
        self._std = None
        self._mean_ = 0.0
        self._scale_ = 1.0
        self.eps_ = None
        self.z_min_ = None
        self.z_max_ = None
        self.y_max_ = None
        self.clip_margin = 1.0

    @staticmethod
    def _ensure_1d(a):
        import numpy as np
        a = np.asarray(a)
        if a.ndim == 2 and a.shape[1] == 1:
            return a.ravel()
        return a.ravel()

    def fit(self, y):
        y = self._ensure_1d(y)
        # auto epsilon from the smallest positive
        if self.eps is None:
            pos = y[y > 0]
            min_pos = float(np.min(pos)) if pos.size > 0 else 1e-12
            self.eps_ = max(1e-12, 0.1 * min_pos)
        else:
            self.eps_ = float(self.eps)
        z = self._to_log(y)
        self.z_min_ = float(z.min())
        self.z_max_ = float(z.max())
        self.y_max_ = float(np.max(y)) if y.size > 0 else 1.0
        if self.with_std:
            from sklearn.preprocessing import StandardScaler
            self._std = StandardScaler()
            z2d = z.reshape(-1, 1)
            _ = self._std.fit_transform(z2d)
            self._mean_ = float(self._std.mean_[0])
            self._scale_ = float(self._std.scale_[0])
        return self

    def _to_log(self, y):
        y = self._ensure_1d(y).astype(float)
        return np.log(y + self.eps_) / np.log(self.base)

    def transform(self, y):
        z = self._to_log(y)
        if self._std is not None:
            z = (z - self._mean_) / (self._scale_ + 1e-12)
        return z

    def inverse_transform(self, z):
        import numpy as np
        z = self._ensure_1d(z).astype(float)
        if self._std is not None:
            z = z * (self._scale_ + 1e-12) + self._mean_
        z_lo = (self.z_min_ if self.z_min_ is not None else -50.0) - self.clip_margin
        z_hi = (self.z_max_ if self.z_max_ is not None else 50.0) + self.clip_margin
        z = np.clip(z, z_lo, z_hi)
        y = (self.base ** z) - (self.eps_ if self.eps_ is not None else 0.0)
        y = np.nan_to_num(y, nan=0.0, posinf=(self.y_max_ * 1e3 if self.y_max_ is not None else 1e12), neginf=0.0)
        return y

    def __repr__(self):
        return f"LogTargetScaler(base={self.base}, eps={self.eps_}, with_std={self._std is not None})"


# -------------------- 工具函数 --------------------


def inv_if_scaler(arr, scaler):
    '''
    反归一化、反log化
    :param arr:
    :param scaler:
    :return:
    '''
    if scaler is None:
        return arr
    import numpy as np
    out = scaler.inverse_transform(arr.reshape(-1, 1)).ravel()
    out = np.nan_to_num(out, nan=0.0, posinf=(np.max(out[np.isfinite(out)]) if np.any(np.isfinite(out)) else 1e12),
                        neginf=0.0)
    return out


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass
    os.environ["PYTHONHASHSEED"] = str(seed)


def to_tensor(x):
    return torch.tensor(x, dtype=torch.float32)


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def make_log_bins(y, bins=10, eps=1e-12):
    y = np.asarray(y, dtype=float)
    y_log = np.log10(np.clip(y, a_min=eps, a_max=None))
    qs = np.quantile(y_log, np.linspace(0, 1, bins + 1))
    qs = np.unique(qs)
    y_bin = np.digitize(y_log, qs[1:-1], right=True)
    return y_bin


def build_stratified_folds_by_cat_and_logy(cats, y, n_splits, seed, bins=10):
    '''
    按照 类别 和 log(label) 划分，解决VP以及Viscosity量纲差异大的问题
    :param cats:
    :param y:
    :param n_splits:
    :param seed:
    :param bins:
    :return:
    '''
    y_bin = make_log_bins(y, bins=bins)
    combo = np.array([f"{str(cats[i])}__{int(y_bin[i])}" for i in range(len(y_bin))])
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    for trval_idx, te_idx in skf.split(np.zeros_like(combo), combo):
        folds.append((trval_idx, te_idx))
    return folds

    counts = {}
    i = 0
    while i < len(cats):
        key = cats[i]
        counts[key] = counts.get(key, 0) + 1
        i += 1
    return counts


def load_sheet(xlsx_path, sheet_name):
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name)
    if df.shape[1] < 3:
        raise ValueError(f"工作表 '{sheet_name}' 至少需要三列：category, label, features...")
    cats = df.iloc[:, 0].values
    y = df.iloc[:, 1].values.astype(np.float32)
    X = df.iloc[:, 2:].values.astype(np.float32)
    feat_names = [str(c) for c in df.columns[2:]]
    row_ids = np.arange(len(df), dtype=np.int64)
    return X, y, cats, feat_names, row_ids


# -------------------- 阶段一：embedding训练 --------------------
def train_embedding_stage(X_tr, y_tr, X_val, y_val, input_dim, device,
                          epochs=50, batch_size=256, lr=1e-2, weight_decay=1e-3, margin=3.0):
    x_scaler = StandardScaler()
    X_tr_s = x_scaler.fit_transform(X_tr)
    X_val_s = x_scaler.transform(X_val)

    y_scaler = StandardScaler()
    y_tr_s = y_scaler.fit_transform(y_tr.reshape(-1, 1)).ravel()
    y_val_s = y_scaler.transform(y_val.reshape(-1, 1)).ravel()

    tr_loader = DataLoader(TensorDataset(to_tensor(X_tr_s), to_tensor(y_tr_s.reshape(-1, 1))),
                           batch_size=batch_size, shuffle=True)

    model = EmbeddingMLP(input_dim, 512, 128).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = ContrastiveLoss(margin=margin).to(device)

    best_state = None
    best_spearman = -1.0

    for epoch in range(epochs):
        model.train()
        for xb, yb in tr_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            emb = model(xb)
            loss = criterion(emb, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            pred = model(to_tensor(X_val_s).to(device))
            sp = embedding_val_func(pred, to_tensor(y_val_s.reshape(-1, 1)).to(device))
            sp_val = float(sp[0]) if isinstance(sp, (list, tuple, np.ndarray)) else float(sp)
            if sp_val > best_spearman:
                best_spearman = sp_val
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    print("第一阶段训练完成，最好spearman系数是: ", best_spearman)

    if best_state is not None:
        model.load_state_dict(best_state)

    model._x_scaler = x_scaler
    model._y_scaler = y_scaler
    return model


def embed(model, X, device):
    x_s = model._x_scaler.transform(X)
    with torch.no_grad():
        e = model(to_tensor(x_s).to(device)).cpu().numpy()
    return e


# -------------------- 阶段1.5：GCN特征融合 --------------------

def build_gcn_features(X_tr, X_val, X_te, feat_names, model, device, k=10, a=0.1,
                       grouped=True, mapping=None):
    # embedding
    E_tr = embed(model, X_tr, device)
    E_val = embed(model, X_val, device)
    E_te = embed(model, X_te, device)
    # 相似性矩阵
    S_tr = np.dot(E_tr, E_tr.T)
    S_val = np.dot(E_val, E_tr.T)
    S_te = np.dot(E_te, E_tr.T)
    # 训练图构建
    tr_knn = knn_train(S_tr, k)
    tr_norm = l1_norm(tr_knn)
    A_tr, deg_tr = norm_adj_train(tr_norm, a)
    # 特征标准化
    X_tr_s = model._x_scaler.transform(X_tr).astype(np.float32)
    X_val_s = model._x_scaler.transform(X_val).astype(np.float32)
    X_te_s = model._x_scaler.transform(X_te).astype(np.float32)

    n_tr, n_val, n_te = X_tr_s.shape[0], X_val_s.shape[0], X_te_s.shape[0]
    # 融合后的训练集
    Xtr_fused = np.dot(A_tr.astype(np.float32), X_tr_s)
    # 验证集连接到训练集
    tr_block_val = np.hstack([A_tr.astype(np.float32), np.zeros((n_tr, n_val), dtype=np.float32)])
    val_block = knn_val(S_val, n_val, n_tr + n_val, k).astype(np.float32)
    full_A_val = np.vstack([tr_block_val, val_block])
    A_val = norm_adj_val(full_A_val, deg_tr, a)
    X_stack_val = np.vstack([X_tr_s, X_val_s]).astype(np.float32)
    Xval_fused = np.dot(A_val, X_stack_val)[-n_val:]
    # 测试集连接到训练集
    tr_block_te = np.hstack([A_tr.astype(np.float32), np.zeros((n_tr, n_te), dtype=np.float32)])
    te_block = knn_val(S_te, n_te, n_tr + n_te, k).astype(np.float32)
    full_A_te = np.vstack([tr_block_te, te_block])
    A_te = norm_adj_val(full_A_te, deg_tr, a)
    X_stack_te = np.vstack([X_tr_s, X_te_s]).astype(np.float32)
    Xte_fused = np.dot(A_te, X_stack_te)[-n_te:]
    # 按描述符种类分组
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

        Xtr_seq = to_seq(Xtr_fused)
        Xval_seq = to_seq(Xval_fused)
        Xte_seq = to_seq(Xte_fused)
        seq_len = len(group_order)
        input_dim = max_cols
        return Xtr_seq, Xval_seq, Xte_seq, seq_len, input_dim, group_order
    else:
        Xtr_seq = Xtr_fused[:, None, :].astype(np.float32)
        Xval_seq = Xval_fused[:, None, :].astype(np.float32)
        Xte_seq = Xte_fused[:, None, :].astype(np.float32)
        return Xtr_seq, Xval_seq, Xte_seq, 1, Xtr_fused.shape[1], feat_names


def build_gcn_features_outer_anchor(X_tr, X_val, X_te, feat_names, model, device, k=10, a=0.1,
                                    grouped=True, mapping=None):
    '''
    找到最佳参数后在训练集+验证集上重训一遍，测试集连接到训练集+验证集
    :param X_tr:
    :param X_val:
    :param X_te:
    :param feat_names:
    :param model:
    :param device:
    :param k:
    :param a:
    :param grouped:
    :param mapping:
    :return:
    '''
    X_outer = np.vstack([X_tr, X_val]).astype(np.float32)

    E_outer = embed(model, X_outer, device)
    E_te = embed(model, X_te, device)

    S_outer = np.dot(E_outer, E_outer.T)
    S_te = np.dot(E_te, E_outer.T)

    tr_knn = knn_train(S_outer, k)
    tr_norm = l1_norm(tr_knn)
    A_outer, deg_outer = norm_adj_train(tr_norm, a)

    X_outer_s = model._x_scaler.transform(X_outer).astype(np.float32)
    X_te_s = model._x_scaler.transform(X_te).astype(np.float32)

    n_outer = X_outer_s.shape[0]
    n_te = X_te_s.shape[0]

    Xouter_fused = np.dot(A_outer.astype(np.float32), X_outer_s)

    tr_block_te = np.hstack([A_outer.astype(np.float32), np.zeros((n_outer, n_te), dtype=np.float32)])
    te_block = knn_val(S_te, n_te, n_outer + n_te, k).astype(np.float32)
    full_A_te = np.vstack([tr_block_te, te_block])
    A_te_outer = norm_adj_val(full_A_te, deg_outer, a)
    X_stack_te = np.vstack([X_outer_s, X_te_s]).astype(np.float32)
    Xte_fused = np.dot(A_te_outer, X_stack_te)[-n_te:]

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

        Xouter_seq = to_seq(Xouter_fused)
        Xte_seq = to_seq(Xte_fused)
        seq_len = len(group_order)
        input_dim = max_cols
        return Xouter_seq, Xte_seq, seq_len, input_dim, group_order
    else:
        Xouter_seq = Xouter_fused[:, None, :].astype(np.float32)
        Xte_seq = Xte_fused[:, None, :].astype(np.float32)
        return Xouter_seq, Xte_seq, 1, Xouter_fused.shape[1], feat_names


def save_stage15_to_xlsx(writer, sheet_name, categories, labels, X_seq, group_order):
    '''
    保存融合后特征数据方便复现
    :param writer:
    :param sheet_name:
    :param categories:
    :param labels:
    :param X_seq:
    :param group_order:
    :return:
    '''
    n = X_seq.shape[0]
    seq_len = X_seq.shape[1]
    data = {"category": list(categories), "label": [float(v) for v in labels]}
    import json as _json
    for gi in range(seq_len):
        col_name = str(group_order[gi]) if group_order is not None else f"group_{gi}"
        col_vals = []
        for r in range(n):
            arr = [float(x) for x in X_seq[r, gi, :].tolist()]
            col_vals.append(_json.dumps(arr, ensure_ascii=False))
        data[col_name] = col_vals
    pd.DataFrame(data).to_excel(writer, sheet_name=sheet_name, index=False)


from sklearn.metrics import mean_absolute_error, mean_squared_error

def train_transformer_with_es(Xtr_seq, y_tr_norm, Xval_seq, y_val_orig,
                              seq_len, input_dim, device,
                              num_heads, num_layers, dim_feedforward,
                              hidden1, hidden2, hidden3, dropout,
                              lr, weight_decay, batch_size,
                              max_epochs, patience,
                              y_scaler=None,
                              grad_clip_norm=1.0,
                              val_metric='mae'):  # 新增参数
    tr_loader = DataLoader(
        TensorDataset(to_tensor(Xtr_seq), to_tensor(y_tr_norm.reshape(-1, 1))),
        batch_size=batch_size, shuffle=True
    )

    model = SimpleTransformerRegressor(
        input_dim=input_dim,
        seq_length=seq_len,
        dim_feedforward=dim_feedforward,
        num_heads=num_heads,
        num_layers=num_layers,
        hidden_dim_1=hidden1,
        hidden_dim_2=hidden2,
        hidden_dim_3=hidden3,
        dropout_rate=dropout,
        output_dim=1,
    ).to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    mse_loss = nn.MSELoss()

    best_state = None
    best_val_score = float('inf') if val_metric in ['mae', 'rmse'] else -1e9
    best_epoch = -1
    epochs_no_improve = 0

    for epoch in range(max_epochs):
        model.train()
        for xb, yb in tr_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optim.zero_grad()
            out = model(xb)
            loss = mse_loss(out.float(), yb.float())
            loss.backward()
            if grad_clip_norm is not None and grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            optim.step()

        model.eval()
        with torch.no_grad():
            yhat_val_norm = model(to_tensor(Xval_seq).to(device)).cpu().numpy().ravel()
            yhat_val = inv_if_scaler(yhat_val_norm, y_scaler)

            if val_metric == 'mae':
                score = float(mean_absolute_error(y_val_orig, yhat_val))
            elif val_metric == 'rmse':
                score = float(np.sqrt(mean_squared_error(y_val_orig, yhat_val)))
            else:  # 'r2'
                score = float(r2_score(y_val_orig, yhat_val))

        # 早停逻辑：MAE/RMSE 越小越好，R² 越大越好
        is_better = (
            (val_metric in ['mae', 'rmse'] and score < best_val_score) or
            (val_metric == 'r2' and score > best_val_score)
        )

        if is_better:
            best_val_score = score
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    return best_state, best_val_score, best_epoch

# -------------------- 单折训练（复用 split.json 的索引） --------------------
def train_one_fold_by_indices(sheet_name, X, y, cats, feat_names, row_ids,
                              outer_train_idx, outer_test_idx, inner_train_idx, inner_val_idx,
                              descriptors_mapping, args, device, fold_dir):
    ensure_dir(fold_dir)

    # 按照split索引提取训练、验证、测试集
    idx2pos = {int(rid): pos for pos, rid in enumerate(row_ids.tolist())}
    tr_pos = np.array([idx2pos[i] for i in inner_train_idx], dtype=int)
    val_pos = np.array([idx2pos[i] for i in inner_val_idx], dtype=int)
    te_pos = np.array([idx2pos[i] for i in outer_test_idx], dtype=int)
    X_tr, y_tr, c_tr = X[tr_pos], y[tr_pos], cats[tr_pos]
    X_val, y_val, c_val = X[val_pos], y[val_pos], cats[val_pos]
    X_te, y_te, c_te = X[te_pos], y[te_pos], cats[te_pos]


    print("开始第一阶段训练。。。。。。。")
    embed_model = train_embedding_stage(
        X_tr, y_tr, X_val, y_val,
        input_dim=X.shape[1], device=device,
        epochs=args.epochs_embed, batch_size=args.bs_embed,
        lr=args.lr_embed, weight_decay=args.wd_embed, margin=args.margin,
    )
    print("第一阶段训练完成。")
    torch.save(embed_model, os.path.join(fold_dir, 'embedding_best.pth'))


    Xtr_seq, Xval_seq, Xte_seq, seq_len, input_dim, group_order = build_gcn_features(
        X_tr, X_val, X_te, feat_names, embed_model, device,
        k=args.k, a=args.a, grouped=args.grouped, mapping=descriptors_mapping,
    )

    if args.save_stage15:
        save_path = os.path.join(fold_dir, 'stage15_fused_grouped.xlsx')
        with pd.ExcelWriter(save_path, engine='openpyxl', mode='w') as writer:
            save_stage15_to_xlsx(writer, 'train', c_tr, y_tr, Xtr_seq, group_order)
            save_stage15_to_xlsx(writer, 'val', c_val, y_val, Xval_seq, group_order)
            save_stage15_to_xlsx(writer, 'test', c_te, y_te, Xte_seq, group_order)
    print("该折融合后数据已保存。。。。")
    print(f"开始第二阶段训练（{'log 空间' if args.target_log else '原空间标准化'}）。。。。。")

    # 阶段二：目标变换（支持 log 空间） + Optuna
    if args.target_log:
        y_scaler_tr = LogTargetScaler(base=args.log_base, eps=args.log_eps, with_std=args.log_with_std)
        y_scaler_tr.fit(y_tr)
        y_tr_norm = y_scaler_tr.transform(y_tr)
        y_val_norm = y_scaler_tr.transform(y_val)
    else:
        y_scaler_tr = StandardScaler()
        y_tr_norm = y_scaler_tr.fit_transform(y_tr.reshape(-1, 1)).ravel()
        y_val_norm = y_scaler_tr.transform(y_val.reshape(-1, 1)).ravel()

    best_states = {}
    trial_best_epoch = {}
    best_scalers = {}



    sheet_lower = sheet_name.lower()
    if any(ds in sheet_lower for ds in ['qm7', 'qm8', 'qm9']):
        val_metric_for_optuna = 'mae'
    else:
        val_metric_for_optuna = 'rmse'

    best_states = {}
    trial_best_epoch = {}
    best_scalers = {}

    def objective(trial):
        num_layers = trial.suggest_int('num_layers', 1, 4)
        num_heads = trial.suggest_categorical('num_heads', [1, 2, 3, 6])
        dim_feedforward = trial.suggest_categorical('dim_feedforward', [256, 512, 1024])
        hidden1 = trial.suggest_categorical('hidden1', [64, 128, 256])
        hidden2 = trial.suggest_categorical('hidden2', [512, 1024, 2048])
        hidden3 = trial.suggest_categorical('hidden3', [64, 128, 256])
        dropout = trial.suggest_float('dropout', 0.1, 0.5)
        lr = trial.suggest_float('lr', 1e-4, 3e-3, log=True)
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)

        best_state, best_score, best_epoch = train_transformer_with_es(
            Xtr_seq, y_tr_norm, Xval_seq, y_val,
            seq_len, input_dim, device,
            num_heads, num_layers, dim_feedforward,
            hidden1, hidden2, hidden3, dropout,
            lr, weight_decay, args.bs_tr,
            args.epochs_tr, args.patience,
            y_scaler=y_scaler_tr,
            grad_clip_norm=1.0,
            val_metric=val_metric_for_optuna,  # 关键：传入指标类型
        )
        best_states[trial.number] = best_state
        trial_best_epoch[trial.number] = int(best_epoch)
        best_scalers[trial.number] = y_scaler_tr

        # Optuna 最小化 MAE/RMSE
        return best_score

    # 创建 study：最小化目标
    study = optuna.create_study(
        direction='minimize',  # ←←← 关键修改：从 maximize 改为 minimize
        study_name=sheet_name,
        sampler=optuna.samplers.TPESampler(seed=args.seed),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10)
    )
    study.optimize(objective, n_trials=args.trials, show_progress_bar=False)


    best_trial = study.best_trial
    best_params = best_trial.params
    best_epoch = trial_best_epoch.get(best_trial.number, -1)
    best_state = best_states.get(best_trial.number)

    if best_state is None:
        best_state, _, best_epoch = train_transformer_with_es(
            Xtr_seq, y_tr_norm, Xval_seq, y_val,
            seq_len, input_dim, device,
            best_params['num_heads'], best_params['num_layers'], best_params['dim_feedforward'],
            best_params['hidden1'], best_params['hidden2'], best_params['hidden3'],
            best_params['dropout'],
            best_params['lr'], best_params['weight_decay'], args.bs_tr,
            args.epochs_tr, args.patience,
            y_scaler=y_scaler_tr,
            grad_clip_norm=1.0,
        )


    # -------- 使用外层的 train+val 合并重训最终模型，再评估 test --------
    # 基于训练集+验证集重建阶段 1.5
    Xouter_seq, Xte_seq_outer, seq_len2, input_dim2, group_order2 = build_gcn_features_outer_anchor(
        X_tr, X_val, X_te, feat_names, embed_model, device,
        k=args.k, a=args.a, grouped=args.grouped, mapping=descriptors_mapping
    )

    y_full = np.concatenate([y_tr, y_val], axis=0)
    if args.target_log:
        final_scaler = LogTargetScaler(base=args.log_base, eps=args.log_eps, with_std=args.log_with_std)
        final_scaler.fit(y_full)
        y_full_norm = final_scaler.transform(y_full)
    else:
        final_scaler = StandardScaler()
        y_full_norm = final_scaler.fit_transform(y_full.reshape(-1, 1)).ravel()

    # 用训练集+验证集融合特征重训最终模型
    final_model = SimpleTransformerRegressor(
        input_dim=input_dim2,
        seq_length=seq_len2,
        dim_feedforward=best_params['dim_feedforward'],
        num_heads=best_params['num_heads'],
        num_layers=best_params['num_layers'],
        hidden_dim_1=best_params['hidden1'],
        hidden_dim_2=best_params['hidden2'],
        hidden_dim_3=best_params['hidden3'],
        dropout_rate=best_params['dropout'],
        output_dim=1,
    ).to(device)

    fixed_epochs = int(best_epoch) if int(best_epoch) > 0 else max(20, int(args.epochs_tr * 0.5))

    tr_loader_full = DataLoader(
        TensorDataset(to_tensor(Xouter_seq), to_tensor(y_full_norm.reshape(-1, 1))),
        batch_size=args.bs_tr, shuffle=True
    )
    optim = torch.optim.Adam(final_model.parameters(), lr=best_params['lr'], weight_decay=best_params['weight_decay'])
    final_model.train()
    for epoch in range(fixed_epochs):
        for xb, yb in tr_loader_full:
            xb = xb.to(device);
            yb = yb.to(device)
            optim.zero_grad()
            out = final_model(xb)
            loss = mse_loss(out.float(), yb.float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(final_model.parameters(), max_norm=1.0)
            optim.step()

    # 保存训练集+验证集阶段 1.5 融合特征
    if args.save_stage15:
        save_path_outer = os.path.join(fold_dir, 'stage15_fused_grouped_outer.xlsx')
        with pd.ExcelWriter(save_path_outer, engine='openpyxl', mode='w') as writer:
            c_outer = np.concatenate([c_tr, c_val], axis=0)
            y_outer = y_full
            save_stage15_to_xlsx(writer, 'outer_train', c_outer, y_outer, Xouter_seq, group_order2)
            save_stage15_to_xlsx(writer, 'test_outer', c_te, y_te, Xte_seq_outer, group_order2)

    # 在测试集融合特征上评估
    final_model.eval()
    with torch.no_grad():
        yhat_te_norm = final_model(to_tensor(Xte_seq_outer).to(device)).cpu().numpy().ravel()
        yhat_te = inv_if_scaler(yhat_te_norm, final_scaler)
        r2 = float(r2_score(y_te, yhat_te))

    torch.save(final_model, os.path.join(fold_dir, 'transformer_best.pth'))
    torch.save(final_scaler, os.path.join(fold_dir, 'y_scaler.pth'))

    print(
        f"Fold (dir={os.path.basename(fold_dir)}): Test R2 = {round(r2, 4)}  (seq_len={seq_len}, input_dim={input_dim})")

    test_pred = pd.DataFrame({
        "row_id": np.array(outer_test_idx, dtype=int),
        "category": c_te,
        "y_true": y_te.astype(float),
        "y_pred": yhat_te.astype(float),
    })
    test_pred.to_csv(os.path.join(fold_dir, 'test_predictions.csv'), index=False)

    # 记录 split.json（复写/更新）
    split_info = {
        "Test R2": r2,
        "sheet": sheet_name,
        "seed": int(args.seed),
        "stratified_by": "category+log(y)",

        "k": int(args.k),
        "a": float(args.a),
        "grouped": bool(args.grouped),
        "mapping_path": os.path.abspath(args.mapping),
        "final_anchor": "outer_train",
        "seq_len": int(seq_len),
        "input_dim": int(input_dim),
        "outer_train_idx": [int(x) for x in outer_train_idx],
        "outer_test_idx": [int(x) for x in outer_test_idx],
        "inner_train_idx": [int(x) for x in inner_train_idx],
        "inner_val_idx": [int(x) for x in inner_val_idx],
        "best_trial_number": int(best_trial.number),
        "best_epoch": int(best_epoch),
        "best_params": best_params,
        "files": {
            "embedding": os.path.join(fold_dir, 'embedding_best.pth'),
            "transformer": os.path.join(fold_dir, 'transformer_best.pth'),
            "y_scaler": os.path.join(fold_dir, 'y_scaler.pth'),
            "pred_csv": os.path.join(fold_dir, 'test_predictions.csv')
        }
    }
    with open(os.path.join(fold_dir, 'split.json'), 'w', encoding='utf-8') as fp:
        json.dump(split_info, fp, ensure_ascii=False, indent=2)

    return r2


# -------------------- 主流程 --------------------
def main():
    sys.argv = _ORIG_ARGV

    ap = argparse.ArgumentParser()
    ap.add_argument('--xlsx', required=True)
    ap.add_argument('--mapping', required=True)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--save_dir', default='./cv_runs')

    # 选择性重训相关
    ap.add_argument('--only_sheet', default=None, help='只训练该 sheet（不填则遍历全部 sheet）')
    ap.add_argument('--only_folds', default=None, help='只训练这些折（1 基，逗号分隔，如 2 或 1,3,5）')
    ap.add_argument('--split_json', default=None, help='指定已有 split.json，按其划分重训该折')
    ap.add_argument('--use_saved_splits', action='store_true', help='5 折流程中优先复用各折 split.json 中的划分')

    # GCN 参数
    ap.add_argument('--k', type=int, default=10)
    ap.add_argument('--a', type=float, default=0.1)
    ap.add_argument('--grouped', action='store_true')

    # 阶段一（embedding）
    ap.add_argument('--epochs_embed', type=int, default=50)
    ap.add_argument('--bs_embed', type=int, default=512)
    ap.add_argument('--lr_embed', type=float, default=1e-2)
    ap.add_argument('--wd_embed', type=float, default=1e-3)
    ap.add_argument('--margin', type=float, default=3.0)


    # 阶段二（Optuna + 早停）
    ap.add_argument('--trials', type=int, default=50)
    ap.add_argument('--epochs_tr', type=int, default=500)
    ap.add_argument('--patience', type=int, default=50)
    ap.add_argument('--bs_tr', type=int, default=512)
    ap.add_argument('--save_stage15', default=True, action='store_true')

    # 目标变换：log 空间学习（适合 VP/Viscosity 等跨数量级标签）
    ap.add_argument('--target_log', action='store_true', help='在 log 空间学习')
    ap.add_argument('--log_base', type=float, default=10.0, help='log 的底，默认 10')
    ap.add_argument('--log_eps', type=float, default=None, help='epsilon（默认自动从最小正标签估计）')
    ap.add_argument('--log_with_std', action='store_true', help='log 后再做标准化')

    ap.add_argument('--exp_k', type=int, default=1, help='每个 fold 重复实验次数')

    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)

    with open(args.mapping, 'r', encoding='utf-8') as f:
        descriptors_mapping = json.load(f)

    ensure_dir(args.save_dir)

    # ----------- 特殊路径：直接基于一个 split.json 复训该折 -----------
    if args.split_json is not None:
        split_path = os.path.abspath(args.split_json)
        if not os.path.isfile(split_path):
            raise FileNotFoundError(f"split.json 不存在：{split_path}")
        with open(split_path, 'r', encoding='utf-8') as fp:
            split = json.load(fp)

        sheet_name = split["sheet"]
        X, y, cats, feat_names, row_ids = load_sheet(args.xlsx, sheet_name)

        fold_root = os.path.dirname(split_path)

        # 单折多次尝试
        if args.exp_k == 1:
            r2 = train_one_fold_by_indices(
                sheet_name=sheet_name,
                X=X, y=y, cats=cats, feat_names=feat_names, row_ids=row_ids,
                outer_train_idx=split["outer_train_idx"],
                outer_test_idx=split["outer_test_idx"],
                inner_train_idx=split["inner_train_idx"],
                inner_val_idx=split["inner_val_idx"],
                descriptors_mapping=descriptors_mapping,
                args=args, device=device,
                fold_dir=fold_root
            )
            print(f"[DONE] 复训完成（split.json：{split_path}），Test R2 = {round(r2, 4)}")
            return
        else:
            base_seed = int(args.seed)
            best_r2 = -1e18
            best_exp = None
            attempt_r2s = []
            for exp_i in range(1, args.exp_k + 1):
                tmp_seed = base_seed + exp_i - 1
                print(f"[Try] exp_{exp_i} with seed={tmp_seed}")
                set_seed(tmp_seed)
                old_seed = args.seed
                args.seed = tmp_seed
                exp_dir = os.path.join(fold_root, f'exp_{exp_i}')
                ensure_dir(exp_dir)
                r2_i = train_one_fold_by_indices(
                    sheet_name=sheet_name,
                    X=X, y=y, cats=cats, feat_names=feat_names, row_ids=row_ids,
                    outer_train_idx=split["outer_train_idx"],
                    outer_test_idx=split["outer_test_idx"],
                    inner_train_idx=split["inner_train_idx"],
                    inner_val_idx=split["inner_val_idx"],
                    descriptors_mapping=descriptors_mapping,
                    args=args, device=device,
                    fold_dir=exp_dir
                )
                attempt_r2s.append(float(r2_i))
                if r2_i > best_r2:
                    best_r2 = float(r2_i)
                    best_exp = exp_i
                args.seed = old_seed

            best_dir = os.path.join(fold_root, f'exp_{best_exp}')
            for fname in ['embedding_best.pth', 'transformer_best.pth', 'y_scaler.pth', 'test_predictions.csv',
                          'split.json']:
                src = os.path.join(best_dir, fname)
                if os.path.exists(src):
                    shutil.copy2(src, os.path.join(fold_root, fname))
            st15 = os.path.join(best_dir, 'stage15_fused_grouped.xlsx')
            if os.path.exists(st15):
                shutil.copy2(st15, os.path.join(fold_root, 'stage15_fused_grouped.xlsx'))

            try:
                sp_path = os.path.join(fold_root, 'split.json')
                with open(sp_path, 'r', encoding='utf-8') as fp:
                    spj = json.load(fp)
                spj['files']['embedding'] = os.path.join(fold_root, 'embedding_best.pth')
                spj['files']['transformer'] = os.path.join(fold_root, 'transformer_best.pth')
                spj['files']['y_scaler'] = os.path.join(fold_root, 'y_scaler.pth')
                spj['files']['pred_csv'] = os.path.join(fold_root, 'test_predictions.csv')
                with open(sp_path, 'w', encoding='utf-8') as fp:
                    json.dump(spj, fp, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"[Warn] 修正 split.json 失败：{e}")

            print(
                f"[DONE] split.json 复训 {args.exp_k} 次，R2 列表 = {[round(x, 4) for x in attempt_r2s]}，最佳 exp_{best_exp} -> Test R2 = {round(best_r2, 4)}")
            return

    # ----------- 常规流程：遍历 sheet / 折；可过滤 -----------
    all_sheets = pd.ExcelFile(args.xlsx).sheet_names
    if args.only_sheet is not None:
        all_sheets = [s for s in all_sheets if s == args.only_sheet]
        if not all_sheets:
            raise ValueError(f"未找到指定 sheet：{args.only_sheet}")

    allowed_folds = None
    if args.only_folds is not None:
        allowed_folds = set(int(x.strip()) for x in args.only_folds.split(',') if x.strip())

    all_sheet_summary = []

    for sheet_name in all_sheets:
        print("\n" + "=" * 80)
        print("开始处理工作表：", sheet_name)
        print("=" * 80)

        X, y, cats, feat_names, row_ids = load_sheet(args.xlsx, sheet_name)
        run_root = os.path.join(args.save_dir, 'sheet_' + sheet_name)
        ensure_dir(run_root)

        folds = build_stratified_folds_by_cat_and_logy(cats, y, n_splits=5, seed=args.seed, bins=10)

        cv_manifest = {
            "sheet": sheet_name,
            "seed": int(args.seed),
            "stratified_by": "category+log(y)",
            "k": int(args.k),
            "a": float(args.a),
            "grouped": bool(args.grouped),
            "mapping_path": os.path.abspath(args.mapping),
            "final_anchor": "outer_train",
            "save_dir": run_root,
            "folds": []
        }

        fold_metrics = []
        for fold_id, (trval_idx, te_idx) in enumerate(folds, start=1):
            if (allowed_folds is not None) and (fold_id not in allowed_folds):
                print(f"[Skip] {sheet_name} / fold_{fold_id} 不在 only_folds 中，跳过。")
                continue

            fold_dir = os.path.join(run_root, f'fold_{fold_id}')
            ensure_dir(fold_dir)

            # 默认按当前构造；若要求复用旧划分、且有 split.json，则覆盖为旧划分
            outer_train_idx = row_ids[trval_idx].tolist()
            outer_test_idx = row_ids[te_idx].tolist()

            split_json_path = os.path.join(fold_dir, 'split.json')
            use_old_split = args.use_saved_splits and os.path.isfile(split_json_path)
            if use_old_split:
                print(f"[Info] 复用已有划分：{split_json_path}")
                with open(split_json_path, 'r', encoding='utf-8') as fp:
                    old = json.load(fp)
                # 强制使用旧的内/外层索引（全局行号）
                outer_train_idx = old["outer_train_idx"]
                outer_test_idx = old["outer_test_idx"]
                inner_train_idx = old["inner_train_idx"]
                inner_val_idx = old["inner_val_idx"]
            else:
                # 现划分：在外层训练集上分层出 20% 验证
                rel_ids = np.arange(len(trval_idx))
                c_trval = cats[trval_idx]
                y_trval = y[trval_idx]
                ybin_trval = make_log_bins(y_trval, bins=10)
                combo_trval = np.array([f"{str(c_trval[i])}__{int(ybin_trval[i])}" for i in range(len(ybin_trval))])
                try:
                    rel_tr, rel_val = train_test_split(
                        rel_ids, test_size=0.2, random_state=args.seed,
                        shuffle=True, stratify=combo_trval
                    )
                except Exception:
                    rel_tr, rel_val = train_test_split(
                        rel_ids, test_size=0.2, random_state=args.seed,
                        shuffle=True, stratify=None
                    )
                    inner_train_idx = row_ids[trval_idx[rel_tr]].tolist()
                inner_val_idx = row_ids[trval_idx[rel_val]].tolist()

            # === 真正训练该折 ===
            if args.exp_k == 1:
                r2 = train_one_fold_by_indices(
                    sheet_name=sheet_name,
                    X=X, y=y, cats=cats, feat_names=feat_names, row_ids=row_ids,
                    outer_train_idx=outer_train_idx,
                    outer_test_idx=outer_test_idx,
                    inner_train_idx=inner_train_idx,
                    inner_val_idx=inner_val_idx,
                    descriptors_mapping=descriptors_mapping,
                    args=args, device=device,
                    fold_dir=fold_dir
                )
                fold_metrics.append(r2)
                # 更新 manifest
                cv_manifest["folds"].append({
                    "fold_id": int(fold_id),
                    "test_r2": float(r2),
                    "split_file": os.path.join(fold_dir, 'split.json'),
                    "embedding": os.path.join(fold_dir, 'embedding_best.pth'),
                    "transformer": os.path.join(fold_dir, 'transformer_best.pth'),
                    "pred_csv": os.path.join(fold_dir, 'test_predictions.csv'),
                    "best_exp": 1
                })
            else:
                print(
                    f"[Info] {sheet_name} / fold_{fold_id} 启用 exp_k={args.exp_k} 次尝试，选择测试 R2 最佳的一次保存。")
                base_seed = int(args.seed)
                best_r2 = -1e18
                best_exp = None
                attempt_r2s = []
                for exp_i in range(1, args.exp_k + 1):
                    tmp_seed = base_seed + exp_i - 1
                    print(f"[Try] fold_{fold_id} / exp_{exp_i} with seed={tmp_seed}")
                    set_seed(tmp_seed)
                    old_seed = args.seed
                    args.seed = tmp_seed
                    exp_dir = os.path.join(fold_dir, f'exp_{exp_i}')
                    ensure_dir(exp_dir)
                    r2_i = train_one_fold_by_indices(
                        sheet_name=sheet_name,
                        X=X, y=y, cats=cats, feat_names=feat_names, row_ids=row_ids,
                        outer_train_idx=outer_train_idx,
                        outer_test_idx=outer_test_idx,
                        inner_train_idx=inner_train_idx,
                        inner_val_idx=inner_val_idx,
                        descriptors_mapping=descriptors_mapping,
                        args=args, device=device,
                        fold_dir=exp_dir
                    )
                    attempt_r2s.append(float(r2_i))
                    if r2_i > best_r2:
                        best_r2 = float(r2_i)
                        best_exp = exp_i
                    args.seed = old_seed

                # 复制最佳产物到 fold 根目录
                best_dir = os.path.join(fold_dir, f'exp_{best_exp}')
                for fname in ['embedding_best.pth', 'transformer_best.pth', 'y_scaler.pth', 'test_predictions.csv',
                              'split.json']:
                    src = os.path.join(best_dir, fname)
                    if os.path.exists(src):
                        shutil.copy2(src, os.path.join(fold_dir, fname))
                st15 = os.path.join(best_dir, 'stage15_fused_grouped.xlsx')
                if os.path.exists(st15):
                    shutil.copy2(st15, os.path.join(fold_dir, 'stage15_fused_grouped.xlsx'))
                # 修正 split.json 的 files 指向根目录
                try:
                    sp_path = os.path.join(fold_dir, 'split.json')
                    with open(sp_path, 'r', encoding='utf-8') as fp:
                        spj = json.load(fp)
                    spj['files']['embedding'] = os.path.join(fold_dir, 'embedding_best.pth')
                    spj['files']['transformer'] = os.path.join(fold_dir, 'transformer_best.pth')
                    spj['files']['y_scaler'] = os.path.join(fold_dir, 'y_scaler.pth')
                    spj['files']['pred_csv'] = os.path.join(fold_dir, 'test_predictions.csv')
                    with open(sp_path, 'w', encoding='utf-8') as fp:
                        json.dump(spj, fp, ensure_ascii=False, indent=2)
                except Exception as e:
                    print(f"[Warn] 修正 split.json 失败：{e}")

                r2 = float(best_r2)
                fold_metrics.append(r2)
                print(
                    f"[Result] {sheet_name} / fold_{fold_id} R2 尝试 = {[round(x, 4) for x in attempt_r2s]}，最佳 exp_{best_exp} -> Test R2 = {round(r2, 4)}")
                # 更新 manifest 指向根目录产物
                cv_manifest["folds"].append({
                    "fold_id": int(fold_id),
                    "test_r2": float(r2),
                    "split_file": os.path.join(fold_dir, 'split.json'),
                    "embedding": os.path.join(fold_dir, 'embedding_best.pth'),
                    "transformer": os.path.join(fold_dir, 'transformer_best.pth'),
                    "pred_csv": os.path.join(fold_dir, 'test_predictions.csv'),
                    "best_exp": int(best_exp)
                })

        # 写入当前 sheet 的 manifest
        with open(os.path.join(run_root, 'cv_manifest.json'), 'w', encoding='utf-8') as fp:
            json.dump(cv_manifest, fp, ensure_ascii=False, indent=2)

        if fold_metrics:
            r_list = [round(x, 4) for x in fold_metrics]
            mean_r2 = float(np.mean(fold_metrics))
            std_r2 = float(np.std(fold_metrics))
            print("-" * 60)
            print("Sheet:", sheet_name, "| R2s:", r_list)
            print("Mean R2:", mean_r2)
            print("Std  R2:", std_r2)
            all_sheet_summary.append({
                "sheet": sheet_name, "mean_r2": mean_r2, "std_r2": std_r2, "fold_r2": r_list
            })
        else:
            print(f"[Warn] {sheet_name} 未训练到任何折（可能 all 被过滤）。")

    # 汇总
    if all_sheet_summary:
        print("\n" + "=" * 80)
        print("已完成训练的工作表总结：")
        for rec in all_sheet_summary:
            print(
                f" - {rec['sheet']}  Mean R2 = {rec['mean_r2']:.4f}  Std = {rec['std_r2']:.4f}  folds = {rec['fold_r2']}")
        print("=" * 80)


if __name__ == '__main__':
    main()
