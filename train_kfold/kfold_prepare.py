import numpy as np
import pandas as pd


def make_log_bins(y, bins=10, eps=1e-12):
    y = np.asarray(y, dtype=float)
    y_log = np.log10(np.clip(y, a_min=eps, a_max=None))
    qs = np.quantile(y_log, np.linspace(0, 1, bins + 1))
    qs = np.unique(qs)
    y_bin = np.digitize(y_log, qs[1:-1], right=True)
    return y_bin




def build_stratified_folds_by_cat_and_logy(cats, y, n_splits, seed, bins=10):

    y_bin = make_log_bins(y, bins=bins)
    combo = np.array([f"{str(cats[i])}__{int(y_bin[i])}" for i in range(len(y_bin))])
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    for trval_idx, te_idx in skf.split(np.zeros_like(combo), combo):
        print("Length of training set and test set: ", len(trval_idx), len(te_idx))
        folds.append((trval_idx, te_idx))
    return folds


def load_sheet(xlsx_path, sheet_name):
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name)
    if df.shape[1] < 3:
        raise ValueError("At least three columns are required: category, label, features...")
    cats = df.iloc[:, 0].values
    y = df.iloc[:, 1].values.astype(np.float32)
    X = df.iloc[:, 2:].values.astype(np.float32)
    feat_names = [str(c) for c in df.columns[2:]]
    row_ids = np.arange(len(df), dtype=np.int64)
    return X, y, cats, feat_names, row_ids


