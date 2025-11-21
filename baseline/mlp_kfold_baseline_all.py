
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
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# ----------------------- Utils -----------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)


def device_from_args(use_gpu: bool) -> torch.device:
    if use_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


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


# ----------------------- Model -----------------------

class MLP(nn.Module):
    def __init__(self, input_dim: int, h1: int, h2: int, drop1: float, drop2: float):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, h1)
        self.do1 = nn.Dropout(drop1)
        self.fc2 = nn.Linear(h1, h2)
        self.do2 = nn.Dropout(drop2)
        self.out = nn.Linear(h2, 1)
        # He init
        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.out.weight, nonlinearity='relu')

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.do1(x)
        x = torch.relu(self.fc2(x))
        x = self.do2(x)
        return self.out(x)


# ----------------------- Train / Eval -----------------------

def train_one(
    device: torch.device,
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    trial,
    max_epochs: int,
    patience: int,
    input_dim: int
):

    h1 = trial.suggest_categorical('h1', [64, 128, 256, 512])
    h2 = trial.suggest_categorical('h2', [32, 64, 128, 256])
    drop1 = trial.suggest_float('drop1', 0.0, 0.5)
    drop2 = trial.suggest_float('drop2', 0.0, 0.5)
    batch_size = trial.suggest_categorical('batch_size', [64, 128, 256])
    lr = trial.suggest_float('lr', 1e-4, 3e-3, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-8, 1e-3, log=True)


    model = MLP(input_dim, h1, h2, drop1, drop2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()


    tr_dataset = TensorDataset(
        torch.from_numpy(X_tr).float(),
        torch.from_numpy(y_tr.reshape(-1, 1)).float()
    )
    val_dataset = TensorDataset(
        torch.from_numpy(X_val).float(),
        torch.from_numpy(y_val.reshape(-1, 1)).float()
    )

    tr_loader = DataLoader(tr_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=4096, shuffle=False, drop_last=False)


    best_state = None
    best_val_r2 = -1e9
    best_epoch = -1
    patience_ctr = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        for xb, yb in tr_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()


        model.eval()
        with torch.no_grad():
            ys = []
            ps = []
            val_loss = 0.0
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                p = model(xb)
                val_loss += loss_fn(p, yb).item() * xb.size(0)
                ys.append(yb.detach().cpu().numpy())
                ps.append(p.detach().cpu().numpy())
            ys = np.concatenate(ys, axis=0).reshape(-1)
            ps = np.concatenate(ps, axis=0).reshape(-1)
            val_mse = val_loss / len(val_dataset)
            val_r2 = r2_score(ys, ps)


        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            best_epoch = epoch
            patience_ctr = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_ctr += 1


        trial.report(best_val_r2, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

        if patience_ctr >= patience:
            break

    return best_val_r2, best_epoch, best_state, {
        "h1": h1, "h2": h2, "drop1": drop1, "drop2": drop2,
        "batch_size": batch_size, "lr": lr, "weight_decay": weight_decay,
        "max_epochs": max_epochs, "patience": patience
    }


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


# ----------------------- Orchestration -----------------------

def run_one_sheet(args, device: torch.device, xlsx_path: str, sheet: str, split_dir_for_sheet: str):
    X, y, cats, feat_names, row_ids = load_sheet(xlsx_path, sheet)

    base_root = os.path.join(args.save_dir, f"sheet_{sheet}")
    run_root = os.path.join(base_root, "mlp")
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
        "model": "MLP(PyTorch)",
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

        input_dim = X_tr_s.shape[1]

        def objective(trial):
            best_val_r2, best_epoch, best_state, hparams = train_one(
                device, X_tr_s, y_tr, X_val_s, y_val, trial,
                max_epochs=args.max_epochs,
                patience=args.patience,
                input_dim=input_dim
            )

            trial.set_user_attr("best_val_r2", float(best_val_r2))
            trial.set_user_attr("best_epoch", int(best_epoch))

            return float(best_val_r2)

        pruner = optuna.pruners.MedianPruner(n_startup_trials=max(5, args.trials//5), n_warmup_steps=50)
        study = optuna.create_study(direction='maximize', study_name=f"{sheet}_mlp_fold_{fold_id}", pruner=pruner)
        study.optimize(objective, n_trials=args.trials, show_progress_bar=False)

        best_params = study.best_trial.params


        class DummyTrial:
            def __init__(self, params): self.params=params
            def suggest_categorical(self,*a,**k): key=a[0]; return self.params[key]
            def suggest_float(self,*a,**k): key=a[0]; return self.params[key]
            def report(self,*a,**k): pass
            def should_prune(self): return False
        dt = DummyTrial(best_params)
        best_val_r2, best_epoch, best_state, hparams = train_one(
            device, X_tr_s, y_tr, X_val_s, y_val, dt,
            max_epochs=args.max_epochs, patience=args.patience, input_dim=input_dim
        )


        model = MLP(input_dim, hparams["h1"], hparams["h2"], hparams["drop1"], hparams["drop2"]).to(device)
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        model.eval()


        with torch.no_grad():
            te_tensor = torch.from_numpy(X_te_s).float().to(device)
            yhat = model(te_tensor).detach().cpu().numpy().reshape(-1)

        r2 = float(r2_score(y_te, yhat))
        mae = float(mean_absolute_error(y_te, yhat))
        mse = float(mean_squared_error(y_te, yhat))
        fold_metrics.append(r2)
        print(f"[{sheet}] Fold {fold_id}: Test R2 = {r2:.4f}")


        model_path = os.path.join(fold_dir, "mlp_best.pt")
        torch.save({
            "model_state": {k: v.cpu() for k, v in best_state.items()},
            "input_dim": int(input_dim),
            "hparams": hparams
        }, model_path)

        scaler_path = os.path.join(fold_dir, "scaler.json")
        save_scaler_json(scaler_path, scaler)

        test_pred = pd.DataFrame({
            "row_id": np.array(outer_test_idx, dtype=int),
            "category": c_te,
            "y_true": y_te.astype(float),
            "y_pred": yhat.astype(float),
        })
        test_pred.to_csv(os.path.join(fold_dir, 'test_predictions.csv'), index=False)

        with open(os.path.join(fold_dir, 'best_params.json'), 'w', encoding='utf-8') as fp:
            json.dump(best_params, fp, ensure_ascii=False, indent=2)

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
            "best_params": best_params,
            "best_val_r2": float(best_val_r2),
            "best_epoch": int(best_epoch),
            "metrics": {"test_r2": r2, "test_mae": mae, "test_mse": mse},
            "files": {
                "mlp": model_path,
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
            "mlp": model_path,
            "pred_csv": os.path.join(fold_dir, 'test_predictions.csv')
        })

    with open(os.path.join(run_root, 'cv_manifest.json'), 'w', encoding='utf-8') as fp:
        json.dump(cv_manifest, fp, ensure_ascii=False, indent=2)

    return float(np.mean(fold_metrics)), float(np.std(fold_metrics)), [round(v, 4) for v in fold_metrics]


def main():
    sys.argv = _ORIG_ARGV
    ap = argparse.ArgumentParser()
    ap.add_argument('--xlsx', required=True, help='An Excel file containing all the descriptors')
    ap.add_argument('--split_root', required=True, help='The root directory contains multiple subdirectories named "sheet_*", and within each of these directories, there is a file named "fold_i/split.json".')
    ap.add_argument('--save_dir', default='./baseline_runs', help='Output the root directory (each sheet will be written to save_dir/sheet_<name>/mlp/)')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--trials', type=int, default=50)
    ap.add_argument('--max_epochs', type=int, default=2000)
    ap.add_argument('--patience', type=int, default=200)
    ap.add_argument('--gpu', action='store_true')

    args = ap.parse_args()
    set_seed(args.seed)
    device = device_from_args(args.gpu)


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
        mean_r2, std_r2, r2_list = run_one_sheet(args, device, args.xlsx, sheet, split_dir_for_sheet)
        summary.append((sheet, mean_r2, std_r2, r2_list))

    print("=" * 70)
    print("baseline completed（MLP）")
    for sheet, mean_r2, std_r2, r2_list in summary:
        print(f"{sheet:>24}: R2={mean_r2:.4f} ± {std_r2:.4f} | folds={r2_list}")


if __name__ == '__main__':
    main()
