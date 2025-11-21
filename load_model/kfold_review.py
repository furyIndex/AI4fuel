
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
from model.embedding_net import EmbeddingMLP
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


    def inverse_transform(self, z):
        import numpy as np
        z = np.asarray(z).reshape(-1)

        if self._std is not None:
            z = z * (getattr(self, "_scale_", 1.0) + 1e-12) + getattr(self, "_mean_", 0.0)

        z_lo = (self.z_min_ if self.z_min_ is not None else -50.0) - self.clip_margin
        z_hi = (self.z_max_ if self.z_max_ is not None else  50.0) + self.clip_margin
        z = np.clip(z, z_lo, z_hi)

        eps = self.eps_ if getattr(self, "eps_", None) is not None else (self.eps if self.eps is not None else 0.0)
        y = (self.base ** z) - eps

        upper = (self.y_max_ * 1e3) if (self.y_max_ is not None) else 1e12
        y = np.nan_to_num(y, nan=0.0, posinf=upper, neginf=0.0)
        return y






def to_tensor(x):
    return torch.tensor(x, dtype=torch.float32)


def inv_if_scaler(arr, scaler):

    if scaler is None:
        return arr
    import numpy as _np
    out = scaler.inverse_transform(arr.reshape(-1, 1)).ravel()
    out = _np.nan_to_num(out, nan=0.0,
                         posinf=(_np.max(out[_np.isfinite(out)]) if _np.any(_np.isfinite(out)) else 1e12),
                         neginf=0.0)
    return out


def safe_load(path, device):

    try:

        from torch.serialization import safe_globals
        with safe_globals([EmbeddingMLP, SimpleTransformerRegressor, StandardScaler, LogTargetScaler]):
            return torch.load(path, map_location=device)
    except Exception:
        return torch.load(path, map_location=device, weights_only=False)



def load_sheet_flexible(xlsx_path, sheet_name):
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name)
    if df.shape[1] < 2:
        raise ValueError("At least two columns are required：label, features...")

    first_is_numeric = pd.api.types.is_numeric_dtype(df.iloc[:, 0])
    second_is_numeric = pd.api.types.is_numeric_dtype(df.iloc[:, 1]) if df.shape[1] > 1 else False
    if first_is_numeric and (df.shape[1] == 2 or second_is_numeric):

        cats = np.array(["NA"] * len(df))
        y = df.iloc[:, 0].values.astype(np.float32)
        X = df.iloc[:, 1:].values.astype(np.float32)
        feat_names = [str(c) for c in df.columns[1:]]
    else:

        cats = df.iloc[:, 0].values
        y = df.iloc[:, 1].values.astype(np.float32)
        X = df.iloc[:, 2:].values.astype(np.float32)
        feat_names = [str(c) for c in df.columns[2:]]
    row_ids = np.arange(len(df), dtype=np.int64)
    return X, y, cats, feat_names, row_ids


def load_stage15_sheet(xlsx_path, sheet_name):
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name)
    if df.shape[1] < 3:
        raise ValueError("Stage 1.5 sheet requires at least three columns：category, label, descriptors...")
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



def embed(model, X, device):
    x_s = model._x_scaler.transform(X)
    with torch.no_grad():
        e = model(to_tensor(x_s).to(device)).cpu().numpy()
    return e




def main():
    sys.argv = _ORIG_ARGV

    ap = argparse.ArgumentParser()
    ap.add_argument('--fold_dir', required=True,
                    help='Training output directory (including embedding_best.pth / transformer_best.pth / y_scaler.pth / split.json)')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')

    ap.add_argument('--repro_which', default='test', help='Reproduce the assessment of which slice (sheet name)')

    ap.add_argument('--out_csv', default='predictions.csv')





    args = ap.parse_args()
    device = torch.device(args.device)

    embedding_path   = os.path.join(args.fold_dir, 'embedding_best.pth')
    transformer_path = os.path.join(args.fold_dir, 'transformer_best.pth')
    y_scaler_path    = os.path.join(args.fold_dir, 'y_scaler.pth')
    if not os.path.exists(embedding_path):   raise FileNotFoundError(embedding_path)
    if not os.path.exists(transformer_path): raise FileNotFoundError(transformer_path)
    if not os.path.exists(y_scaler_path):    raise FileNotFoundError(y_scaler_path)

    embed_model = safe_load(embedding_path, device);   embed_model.eval()
    trans_model = safe_load(transformer_path, device); trans_model.eval()
    y_scaler    = safe_load(y_scaler_path, device)

    st15_path = os.path.join(args.fold_dir, 'stage15_fused_grouped_outer.xlsx')
    if not os.path.exists(st15_path):
        raise FileNotFoundError("Cannot find the stage 1.5 feature file: " + st15_path)
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



if __name__ == '__main__':
    main()
