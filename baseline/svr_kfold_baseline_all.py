
import sys
_ORIG_ARGV = sys.argv[:]
sys.argv = [sys.argv[0]]

import argparse
import json
import os
import random
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import optuna
import joblib
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler


_CUML_AVAILABLE = False
try:
    from cuml.svm import SVR as cuSVR
    _CUML_AVAILABLE = True
except Exception as e:
    _CUML_AVAILABLE = False

# CPU fallback
from sklearn.svm import SVR as skSVR




def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)


def class_counts(cats: np.ndarray) -> Dict:
    d: Dict = {}
    for k in cats:
        d[k] = d.get(k, 0) + 1
    return d


def load_sheet(xlsx_path: str, sheet_name: str):

    df = pd.read_excel(xlsx_path, sheet_name=sheet_name)
    if df.shape[1] < 3:
        raise ValueError(f"'{sheet_name}' At least three columns are required：category, label, features...")
    cats = df.iloc[:, 0].values
    y = df.iloc[:, 1].values.astype(np.float32)
    X = df.iloc[:, 2:].values.astype(np.float32)
    feat_names = [str(c) for c in df.columns[2:]]
    row_ids = np.arange(len(df), dtype=np.int64)
    return X, y, cats, feat_names, row_ids


def read_split_json(path: str):
    with open(path, 'r', encoding='utf-8') as fp:
        s = json.load(fp)
    def pick(d: dict, names: List[str]):
        for n in names:
            if n in d:
                return d[n]
        raise KeyError(f"The key {names} could not be found in {path}.")
    inner_train = pick(s, ["inner_train_idx", "inner_train_index", "train_inner_idx", "train_idx_inner"])
    inner_val = pick(s, ["inner_val_idx", "inner_valid_idx", "val_inner_idx", "valid_idx_inner"])
    outer_test = pick(s, ["outer_test_idx", "test_outer_idx", "test_idx", "outer_test_index"])
    return np.array(inner_train, dtype=int), np.array(inner_val, dtype=int), np.array(outer_test, dtype=int)


def fit_scaler_minmax(X_tr: np.ndarray) -> MinMaxScaler:
    scaler = MinMaxScaler()
    scaler.fit(X_tr)
    return scaler


def apply_scaler(scaler: MinMaxScaler, X: np.ndarray) -> np.ndarray:
    return scaler.transform(X)


def save_scaler_json(path: str, scaler: MinMaxScaler):
    meta = {
        "feature_range": list(scaler.feature_range) if hasattr(scaler, "feature_range") else [0, 1],
        "data_min_": scaler.data_min_.tolist(),
        "data_max_": scaler.data_max_.tolist(),
        "data_range_": scaler.data_range_.tolist(),
        "n_features_in_": int(scaler.n_features_in_)
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)




class SVRBackend:
    def __init__(self, use_gpu: bool, params: Dict):
        self.use_gpu = (use_gpu and _CUML_AVAILABLE)
        self.params = params
        if self.use_gpu:
            # cuML SVR
            self.model = cuSVR(**params)
        else:
            self.model = skSVR(**params)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def save(self, path: str):

        joblib.dump(self.model, path)




def run_one_sheet(args, xlsx_path: str, sheet: str, split_dir_for_sheet: str):
    X, y, cats, feat_names, row_ids = load_sheet(xlsx_path, sheet)

    base_root = os.path.join(args.save_dir, f"sheet_{sheet}")
    run_root = os.path.join(base_root, "svr")
    ensure_dir(run_root)

    # Gather folds
    fold_splits = []
    for i in range(1, 6):
        sp_path = os.path.join(split_dir_for_sheet, f"fold_{i}", "split.json")
        if not os.path.exists(sp_path):
            raise FileNotFoundError(f"can not find split.json: {sp_path}")
        inner_train_idx, inner_val_idx, outer_test_idx = read_split_json(sp_path)
        fold_splits.append((inner_train_idx, inner_val_idx, outer_test_idx))

    cv_manifest = {
        "sheet": sheet,
        "seed": int(args.seed),
        "stratified_by": "category",
        "save_dir": run_root,
        "model": "SVR (cuML GPU / sklearn CPU)",
        "folds": []
    }

    fold_metrics = []
    for fold_id, (inner_train_idx, inner_val_idx, outer_test_idx) in enumerate(fold_splits, start=1):
        fold_dir = os.path.join(run_root, f"fold_{fold_id}")
        ensure_dir(fold_dir)

        X_tr, y_tr = X[inner_train_idx], y[inner_train_idx]
        X_val, y_val = X[inner_val_idx], y[inner_val_idx]
        X_te, y_te = X[outer_test_idx], y[outer_test_idx]
        c_te = cats[outer_test_idx]

        per_class_all = class_counts(cats)
        per_class_outer_test = class_counts(c_te)


        scaler = fit_scaler_minmax(X_tr)
        X_tr_s = apply_scaler(scaler, X_tr)
        X_val_s = apply_scaler(scaler, X_val)
        X_te_s = apply_scaler(scaler, X_te)


        def objective(trial):
            C = trial.suggest_categorical('C', [1, 10, 100, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000])
            epsilon = trial.suggest_categorical('epsilon', [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])
            gamma_choice = trial.suggest_categorical('gamma_choice', ['scale', 'auto', 'float'])
            if gamma_choice == 'float':
                gamma = trial.suggest_float('gamma', 1e-4, 5e-2, log=True)
            else:
                gamma = gamma_choice

            params = dict(kernel='rbf', C=C, epsilon=epsilon, gamma=gamma, cache_size=2000)
            model = SVRBackend(use_gpu=args.gpu, params=params)
            model.fit(X_tr_s, y_tr)
            yhat_val = model.predict(X_val_s)
            r2 = float(r2_score(y_val, yhat_val))
            return r2

        study = optuna.create_study(direction='maximize', study_name=f"{sheet}_svr_fold_{fold_id}")
        study.optimize(objective, n_trials=args.trials, show_progress_bar=False)
        best_params = study.best_trial.params


        if best_params['gamma_choice'] == 'float':
            gamma = best_params['gamma']
        else:
            gamma = best_params['gamma_choice']
        final_params = dict(kernel='rbf', C=best_params['C'], epsilon=best_params['epsilon'], gamma=gamma, cache_size=2000)


        model = SVRBackend(use_gpu=args.gpu, params=final_params)
        model.fit(X_tr_s, y_tr)


        yhat_te = model.predict(X_te_s)
        r2 = float(r2_score(y_te, yhat_te))
        mae = float(mean_absolute_error(y_te, yhat_te))
        mse = float(mean_squared_error(y_te, yhat_te))
        fold_metrics.append(r2)
        print(f"[{sheet}] Fold {fold_id}: Test R2 = {r2:.4f}")


        model_path = os.path.join(fold_dir, "svr_best.joblib")
        model.save(model_path)

        scaler_path = os.path.join(fold_dir, "scaler.json")
        save_scaler_json(scaler_path, scaler)

        test_pred = pd.DataFrame({
            "row_id": np.array(outer_test_idx, dtype=int),
            "category": c_te,
            "y_true": y_te.astype(float),
            "y_pred": np.asarray(yhat_te).astype(float),
        })
        test_pred.to_csv(os.path.join(fold_dir, 'test_predictions.csv'), index=False)


        to_dump = {k:v for k,v in best_params.items() if k != 'gamma_choice'}
        if 'gamma' not in to_dump:

            to_dump['gamma'] = gamma

        with open(os.path.join(fold_dir, 'best_params.json'), 'w', encoding='utf-8') as fp:
            json.dump(to_dump, fp, ensure_ascii=False, indent=2)

        split_info = {
            "sheet": sheet,
            "seed": int(args.seed),
            "stratified_by": "category",
            "per_class_all": per_class_all,
            "per_class_outer_test": per_class_outer_test,
            "outer_train_idx": [int(x) for x in np.setdiff1d(np.arange(len(X)), outer_test_idx)],
            "outer_test_idx": [int(x) for x in outer_test_idx],
            "inner_train_idx": [int(x) for x in inner_train_idx],
            "inner_val_idx": [int(x) for x in inner_val_idx],
            "best_params": to_dump,
            "metrics": {"test_r2": r2, "test_mae": mae, "test_mse": mse},
            "backend": ("cuML" if (args.gpu and _CUML_AVAILABLE) else "sklearn"),
            "files": {
                "svr": model_path,
                "scaler": scaler_path,
                "pred_csv": os.path.join(fold_dir, 'test_predictions.csv')
            }
        }
        with open(os.path.join(fold_dir, 'split.json'), 'w', encoding='utf-8') as fp:
            json.dump(split_info, fp, ensure_ascii=False, indent=2)

        cv_manifest["folds"].append({
            "fold_id": int(fold_id),
            "test_r2": r2,
            "split_file": os.path.join(fold_dir, 'split.json'),
            "svr": model_path,
            "pred_csv": os.path.join(fold_dir, 'test_predictions.csv')
        })

    with open(os.path.join(run_root, 'cv_manifest.json'), 'w', encoding='utf-8') as fp:
        json.dump(cv_manifest, fp, ensure_ascii=False, indent=2)

    return float(np.mean(fold_metrics)), float(np.std(fold_metrics)), [round(v, 4) for v in fold_metrics]


def main():
    sys.argv = _ORIG_ARGV
    ap = argparse.ArgumentParser()
    ap.add_argument('--xlsx', required=True)
    ap.add_argument('--split_root', required=True)
    ap.add_argument('--save_dir', default='./baseline_runs')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--trials', type=int, default=50)
    ap.add_argument('--gpu', action='store_true')

    args = ap.parse_args()
    set_seed(args.seed)


    all_sheet_dirs = [d for d in sorted(os.listdir(args.split_root)) if d.startswith('sheet_') and os.path.isdir(os.path.join(args.split_root, d))]
    if len(all_sheet_dirs) == 0:
        raise RuntimeError(f"No sheet_* directories were found under the {args.split_root} directory.")
    print(f"find {len(all_sheet_dirs)} sheets：{', '.join(all_sheet_dirs)}")

    xls = pd.ExcelFile(args.xlsx)
    available_sheets = set(xls.sheet_names)

    summary = []
    for d in all_sheet_dirs:
        sheet = d[len('sheet_'):]
        if sheet not in available_sheets:
            continue
        split_dir_for_sheet = os.path.join(args.split_root, d)
        mean_r2, std_r2, r2_list = run_one_sheet(args, args.xlsx, sheet, split_dir_for_sheet)
        summary.append((sheet, mean_r2, std_r2, r2_list))

    print("=" * 72)
    print("baseline completed（SVR）")
    for sheet, mean_r2, std_r2, r2_list in summary:
        print(f"{sheet:>24}: R2={mean_r2:.4f} ± {std_r2:.4f} | folds={r2_list}")


if __name__ == '__main__':
    main()
