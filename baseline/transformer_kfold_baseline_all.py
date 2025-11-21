
import sys
_ORIG_ARGV = sys.argv[:]
sys.argv = [sys.argv[0]]

import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import argparse
import json
import os
import random
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import optuna
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


from model.transformerModel import SimpleTransformerRegressor




def set_seed(seed: int = 42):
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


def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)


def device_from_args(use_gpu: bool) -> torch.device:
    return torch.device("cuda") if (use_gpu and torch.cuda.is_available()) else torch.device("cpu")


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


def save_scaler_json(path: str, scaler: StandardScaler):
    meta = {
        "mean_": scaler.mean_.tolist(),
        "scale_": scaler.scale_.tolist(),
        "var_": scaler.var_.tolist(),
        "n_features_in_": int(scaler.n_features_in_)
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


# ----------------------- Grouping -----------------------

def build_grouped_sequences(X: np.ndarray, feat_names: List[str], mapping: Dict[str, List[str]]):

    group_order = list(mapping.keys())
    max_cols = max((len(mapping[g]) for g in group_order), default=0)
    name2idx = {name: i for i, name in enumerate(feat_names)}

    N = X.shape[0]
    seq_len = len(group_order)
    input_dim = max_cols if max_cols > 0 else 1

    X_seq = np.zeros((N, seq_len, input_dim), dtype=np.float32)

    for gi, g in enumerate(group_order):
        cols = mapping.get(g, [])
        for pi in range(len(cols)):
            if pi >= input_dim:
                break
            col_name = cols[pi]
            if col_name in name2idx:
                X_seq[:, gi, pi] = X[:, name2idx[col_name]].astype(np.float32)
    return X_seq, seq_len, input_dim, group_order




def msle_loss(pred, target):
    pred = torch.clamp(pred, min=1e-7)
    target = torch.clamp(target, min=1e-7)
    log_pred = torch.log1p(pred)
    log_target = torch.log1p(target)
    return torch.mean((log_pred - log_target) ** 2)


def train_transformer_with_es(
        device: torch.device,
        X_tr_seq: np.ndarray, y_tr: np.ndarray,
        X_val_seq: np.ndarray, y_val: np.ndarray,
        seq_len: int, input_dim: int,
        num_heads: int, num_layers: int, dim_feedforward: int,
        hidden1: int, hidden2: int, hidden3: int, dropout: float,
        lr: float, weight_decay: float, batch_size: int,
        max_epochs: int, patience: int, loss_name: str,
        trial=None
):
    tr_ds = TensorDataset(torch.from_numpy(X_tr_seq).float(), torch.from_numpy(y_tr.reshape(-1, 1)).float())
    val_ds = TensorDataset(torch.from_numpy(X_val_seq).float(), torch.from_numpy(y_val.reshape(-1, 1)).float())
    tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=4096, shuffle=False, pin_memory=True)

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
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    mse = nn.MSELoss()

    best_state = None
    best_r2 = -1e9
    best_epoch = -1
    bad = 0

    for epoch in range(1, max_epochs + 1):
        # train
        model.train()
        for xb, yb in tr_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            out = model(xb)
            loss = msle_loss(out, yb) if loss_name == 'msle' else mse(out.float(), yb.float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        # validate
        model.eval()
        with torch.no_grad():
            ys = []
            ps = []
            for xb, yb in val_loader:
                xb = xb.to(device)
                p = model(xb).detach().cpu().numpy()
                ps.append(p)
                ys.append(yb.numpy())
            ys = np.concatenate(ys, axis=0).reshape(-1)
            ps = np.concatenate(ps, axis=0).reshape(-1)
            cur_r2 = float(r2_score(ys, ps))

        if trial is not None:
            trial.report(cur_r2, step=epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        if cur_r2 > best_r2:
            best_r2 = cur_r2
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    return best_state, best_r2, best_epoch




def run_one_sheet(args, device: torch.device, xlsx_path: str, mapping: Dict, sheet: str, split_dir_for_sheet: str):
    X, y, cats, feat_names, row_ids = load_sheet(xlsx_path, sheet)

    base_root = os.path.join(args.save_dir, f"sheet_{sheet}")
    run_root = os.path.join(base_root, "transformer_raw")
    ensure_dir(run_root)


    fold_splits = []
    for i in range(1, 6):
        sp_path = os.path.join(split_dir_for_sheet, f"fold_{i}", "split.json")
        if not os.path.exists(sp_path):
            raise FileNotFoundError(f"找不到 split.json: {sp_path}")
        inner_train_idx, inner_val_idx, outer_test_idx = read_split_json(sp_path)
        fold_splits.append((inner_train_idx, inner_val_idx, outer_test_idx))

    cv_manifest = {
        "sheet": sheet,
        "seed": int(args.seed),
        "grouped": True,
        "mapping_path": os.path.abspath(args.mapping),
        "save_dir": run_root,
        "model": "Transformer(SimpleTransformerRegressor)",
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


        scaler = StandardScaler()
        scaler.fit(X_tr)
        X_tr_s = scaler.transform(X_tr).astype(np.float32)
        X_val_s = scaler.transform(X_val).astype(np.float32)
        X_te_s = scaler.transform(X_te).astype(np.float32)


        Xtr_seq, seq_len, input_dim, group_order = build_grouped_sequences(X_tr_s, feat_names, mapping)
        Xval_seq, _, _, _ = build_grouped_sequences(X_val_s, feat_names, mapping)
        Xte_seq,  _, _, _ = build_grouped_sequences(X_te_s, feat_names, mapping)


        best_states = {}
        trial_best_epoch = {}

        def objective(trial):
            num_layers = trial.suggest_int('num_layers', 1, 4)
            num_heads = trial.suggest_categorical('num_heads', [1, 2])
            dim_feedforward = trial.suggest_categorical('dim_feedforward', [256, 512, 1024])
            hidden1 = trial.suggest_categorical('hidden1', [64, 128, 256])
            hidden2 = trial.suggest_categorical('hidden2', [512, 1024, 2048])
            hidden3 = trial.suggest_categorical('hidden3', [64, 128, 256])
            dropout = trial.suggest_float('dropout', 0.1, 0.6)
            lr = trial.suggest_float('lr', 1e-4, 3e-3, log=True)
            weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
            loss_name = trial.suggest_categorical('loss', ['mse'])

            best_state, best_r2, best_epoch = train_transformer_with_es(
                device, Xtr_seq, y_tr, Xval_seq, y_val, seq_len, input_dim,
                num_heads, num_layers, dim_feedforward,
                hidden1, hidden2, hidden3, dropout,
                lr, weight_decay, args.bs_tr,
                args.epochs_tr, args.patience, loss_name, trial
            )
            best_states[trial.number] = best_state
            trial_best_epoch[trial.number] = int(best_epoch)
            return best_r2

        study = optuna.create_study(direction='maximize', study_name=f"{sheet}_transformer_fold_{fold_id}")
        study.optimize(objective, n_trials=args.trials, show_progress_bar=False)

        best_trial = study.best_trial
        best_params = best_trial.params
        best_epoch = trial_best_epoch.get(best_trial.number, -1)
        best_state = best_states.get(best_trial.number)


        model = SimpleTransformerRegressor(
            input_dim=input_dim,
            seq_length=seq_len,
            dim_feedforward=best_params['dim_feedforward'],
            num_heads=best_params['num_heads'],
            num_layers=best_params['num_layers'],
            hidden_dim_1=best_params['hidden1'],
            hidden_dim_2=best_params['hidden2'],
            hidden_dim_3=best_params['hidden3'],
            dropout_rate=best_params['dropout'],
            output_dim=1,
        ).to(device)
        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()

        with torch.no_grad():
            yhat_te = model(torch.from_numpy(Xte_seq).float().to(device)).cpu().numpy().reshape(-1)

        r2 = float(r2_score(y_te, yhat_te))
        mae = float(mean_absolute_error(y_te, yhat_te))
        mse = float(mean_squared_error(y_te, yhat_te))
        fold_metrics.append(r2)
        print(f"[{sheet}] Fold {fold_id}: Test R2 = {r2:.4f} (seq_len={seq_len}, input_dim={input_dim})")

        # Save artifacts
        model_path = os.path.join(fold_dir, 'transformer_best.pth')
        torch.save({
            "state_dict": {k: v.cpu() for k, v in (best_state or {}).items()},
            "arch": {
                "input_dim": int(input_dim),
                "seq_len": int(seq_len),
                "dim_feedforward": best_params['dim_feedforward'],
                "num_heads": best_params['num_heads'],
                "num_layers": best_params['num_layers'],
                "hidden_dim_1": best_params['hidden1'],
                "hidden_dim_2": best_params['hidden2'],
                "hidden_dim_3": best_params['hidden3'],
                "dropout": best_params['dropout'],
                "output_dim": 1
            }
        }, model_path)

        scaler_path = os.path.join(fold_dir, "scaler.json")
        save_scaler_json(scaler_path, scaler)

        test_pred = pd.DataFrame({
            "row_id": np.array(outer_test_idx, dtype=int),
            "category": c_te,
            "y_true": y_te.astype(float),
            "y_pred": yhat_te.astype(float),
        })
        test_pred.to_csv(os.path.join(fold_dir, 'test_predictions.csv'), index=False)

        with open(os.path.join(fold_dir, 'best_params.json'), 'w', encoding='utf-8') as fp:
            json.dump(best_params, fp, ensure_ascii=False, indent=2)

        split_info = {
            "sheet": sheet,
            "seed": int(args.seed),
            "grouped": True,
            "mapping_path": os.path.abspath(args.mapping),
            "seq_len": int(seq_len),
            "input_dim": int(input_dim),
            "per_class_all": per_class_all,
            "per_class_outer_test": per_class_outer_test,
            "outer_train_idx": [int(x) for x in np.setdiff1d(np.arange(len(X)), outer_test_idx)],
            "outer_test_idx": [int(x) for x in outer_test_idx],
            "inner_train_idx": [int(x) for x in inner_train_idx],
            "inner_val_idx": [int(x) for x in inner_val_idx],
            "best_trial_number": int(best_trial.number),
            "best_epoch": int(best_epoch),
            "best_params": best_params,
            "files": {
                "transformer": model_path,
                "scaler": scaler_path,
                "pred_csv": os.path.join(fold_dir, 'test_predictions.csv')
            }
        }
        with open(os.path.join(fold_dir, 'split.json'), 'w', encoding='utf-8') as fp:
            json.dump(split_info, fp, ensure_ascii=False, indent=2)

        cv_manifest.setdefault("folds", []).append({
            "fold_id": int(fold_id),
            "test_r2": r2,
            "split_file": os.path.join(fold_dir, 'split.json'),
            "transformer": model_path,
            "pred_csv": os.path.join(fold_dir, 'test_predictions.csv')
        })

    with open(os.path.join(run_root, 'cv_manifest.json'), 'w', encoding='utf-8') as fp:
        json.dump(cv_manifest, fp, ensure_ascii=False, indent=2)

    return float(np.mean(fold_metrics)), float(np.std(fold_metrics)), [round(v, 4) for v in fold_metrics]


# ----------------------- Main -----------------------

def main():
    sys.argv = _ORIG_ARGV
    ap = argparse.ArgumentParser()
    ap.add_argument('--xlsx', required=True)
    ap.add_argument('--mapping', required=True)
    ap.add_argument('--split_root', required=True)
    ap.add_argument('--save_dir', default='./baseline_runs')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--trials', type=int, default=30)
    ap.add_argument('--epochs_tr', type=int, default=500)
    ap.add_argument('--patience', type=int, default=50)
    ap.add_argument('--bs_tr', type=int, default=128)
    ap.add_argument('--gpu', action='store_true')

    args = ap.parse_args()
    set_seed(args.seed)
    device = device_from_args(args.gpu)


    with open(args.mapping, 'r', encoding='utf-8') as f:
        descriptors_mapping = json.load(f)


    all_sheet_dirs = [d for d in sorted(os.listdir(args.split_root)) if d.startswith('sheet_') and os.path.isdir(os.path.join(args.split_root, d))]
    if len(all_sheet_dirs) == 0:
        raise RuntimeError(f"No sheet_* directories were found under the {args.split_root} directory.")


    xls = pd.ExcelFile(args.xlsx)
    available_sheets = set(xls.sheet_names)

    summary = []
    for d in all_sheet_dirs:
        sheet = d[len('sheet_'):]
        if sheet not in available_sheets:
            continue
        split_dir_for_sheet = os.path.join(args.split_root, d)
        mean_r2, std_r2, r2_list = run_one_sheet(args, device, args.xlsx, descriptors_mapping, sheet, split_dir_for_sheet)
        summary.append((sheet, mean_r2, std_r2, r2_list))

    print("=" * 72)
    print("baseline completed（Transformer）")
    for sheet, mean_r2, std_r2, r2_list in summary:
        print(f"{sheet:>24}: R2={mean_r2:.4f} ± {std_r2:.4f} | folds={r2_list}")


if __name__ == '__main__':
    main()
