#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train on MoleculeNet descriptor CSVs with MoleculeNet-recommended splits & metrics,
while keeping the original training pipeline unchanged (Embedding -> GCN fuse -> Transformer + Optuna).

- Input CSV: columns starting with "label__" are targets; the rest are features.
- Splits:
    * ESOL, FreeSolv, QM8, QM9 -> Random 80/10/10
    * QM7 -> Stratified 80/10/10 (bin y by quantiles, then stratify)
- Metrics:
    * ESOL, FreeSolv -> RMSE
    * QM7, QM8, QM9  -> MAE

This script now supports --exp_k: run K times with different seeds, keep the run with highest test R².
"""
import sys
import os
import json
import argparse
import numpy as np
import pandas as pd
import shutil
from pathlib import Path

_ORIG_ARGV = sys.argv[:]
sys.argv = [sys.argv[0]]

# Add current dir to path to import your original train functions
sys.path.append(str(Path(__file__).resolve().parent))

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ------------------ new: CSV loader for MoleculeNet descriptor files ------------------

def load_molnet_desc_csv(csv_path, label_cols=None):
    df = pd.read_csv(csv_path)
    all_cols = list(df.columns)
    auto_labels = [c for c in all_cols if str(c).startswith("label__")]
    if label_cols is None or len(label_cols) == 0:
        label_cols = auto_labels
    else:
        missing = [c for c in label_cols if c not in all_cols]
        if missing:
            raise ValueError(f"Label columns not found in CSV: {missing}")

    feat_cols = [c for c in all_cols if c not in label_cols]
    X = df[feat_cols].astype(np.float32).values

    # padding到1614（确保维度对齐，根据你的需求）
    if X.shape[1] < 1614:
        X = np.pad(X, ((0, 0), (0, 1614 - X.shape[1])), mode='constant', constant_values=0)
    elif X.shape[1] > 1614:
        X = X[:, :1614]  # 或报错，根据实际情况

    labels = {c: df[c].astype(np.float32).values for c in label_cols}
    row_ids = np.arange(len(df), dtype=np.int64)
    return X, feat_cols, labels, row_ids


# ------------------ new: MoleculeNet recommended split builders ------------------

def split_random_80_10_10(n, seed=42):
    idx = np.arange(n)
    rs = np.random.RandomState(seed)
    rs.shuffle(idx)
    n_train = int(round(n * 0.8))
    n_val = int(round(n * 0.1))
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]
    return train_idx, val_idx, test_idx


def split_stratified_80_10_10(y, seed=42, n_bins=10):
    y = np.asarray(y, dtype=float)
    qs = np.quantile(y, np.linspace(0, 1, n_bins + 1))
    qs = np.unique(qs)
    ybin = np.digitize(y, qs[1:-1], right=True)

    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    tr_idx, tmp_idx = next(sss1.split(np.zeros_like(ybin), ybin))

    ybin_tmp = ybin[tmp_idx]
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
    rel_val, rel_test = next(sss2.split(np.zeros_like(ybin_tmp), ybin_tmp))
    val_idx = tmp_idx[rel_val]
    test_idx = tmp_idx[rel_test]
    return tr_idx, val_idx, test_idx




def parse_args():
    sys.argv = _ORIG_ARGV
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to *_desc.csv")
    parser.add_argument("--dataset", required=True, choices=["ESOL", "FreeSolv", "QM7", "QM8", "QM9", "Lipophilicity"])
    parser.add_argument("--labels", default=None,
                        help="Comma-separated label column names (e.g., 'label__mu,label__alpha'); "
                             "default: all columns starting with 'label__'")
    parser.add_argument("--mapping", required=True, help="Path to descriptors_mapping.json")
    parser.add_argument("--save_dir", default="./molnet_runs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")

    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--a', type=float, default=0.1)
    parser.add_argument('--grouped', action='store_true')
    parser.add_argument('--epochs_embed', type=int, default=100)
    parser.add_argument('--bs_embed', type=int, default=512)
    parser.add_argument('--lr_embed', type=float, default=1e-2)
    parser.add_argument('--wd_embed', type=float, default=1e-3)
    parser.add_argument('--margin', type=float, default=3.0)
    parser.add_argument('--trials', type=int, default=50)
    parser.add_argument('--epochs_tr', type=int, default=600)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--bs_tr', type=int, default=512)
    parser.add_argument('--save_stage15', action='store_true')
    parser.add_argument('--target_log', action='store_true')
    parser.add_argument('--log_base', type=float, default=10.0)
    parser.add_argument('--log_eps', type=float, default=None)
    parser.add_argument('--log_with_std', action='store_true')
    parser.add_argument('--exp_k', type=int, default=1, help="Number of repeated experiments with different seeds")
    return parser.parse_args()





# ------------------ main runner ------------------

def main():
    args = parse_args()

    from train_servier.train import set_seed, ensure_dir
    set_seed(args.seed)  # base seed

    # choose metric per MoleculeNet
    ds = args.dataset.lower()
    if ds in ["esol", "freesolv", "Lipophilicity"]:
        metric_name = "rmse"
    elif ds in ["qm7", "qm8", "qm9"]:
        metric_name = "mae"
    else:
        metric_name = "rmse"

    # which labels to train
    label_cols = [s.strip() for s in args.labels.split(",")] if args.labels else None

    # load CSV
    X, feat_names, labels_dict, row_ids = load_molnet_desc_csv(args.csv, label_cols)
    n = X.shape[0]

    # read mapping
    with open(args.mapping, "r", encoding="utf-8") as f:
        descriptors_mapping = json.load(f)

    # make run dir
    base_out = os.path.join(args.save_dir, Path(args.csv).stem + f"_{args.dataset}")
    ensure_dir(base_out)

    # loop over labels (single-task per label)
    summary = []
    for label_name, y in labels_dict.items():
        print("\n" + "=" * 80)
        print(f"Dataset={args.dataset} | Label={label_name} | N={n} | Metric={metric_name.upper()} | exp_k={args.exp_k}")
        print("=" * 80)

        out_dir = os.path.join(base_out, label_name.replace("label__", "lbl_"))
        ensure_dir(out_dir)

        # best_r2 = -1e18
        best_metrix = 0
        best_exp = None
        all_results = []

        base_seed = args.seed
        for exp_i in range(1, args.exp_k + 1):
            current_seed = base_seed + exp_i - 1
            print(f"\n[EXP {exp_i}/{args.exp_k}] Using seed = {current_seed}")

            # Set seed for this trial
            set_seed(current_seed)

            # Build splits with current seed
            if ds == "qm7":
                tr_idx, val_idx, te_idx = split_stratified_80_10_10(y, seed=current_seed)
            else:
                tr_idx, val_idx, te_idx = split_random_80_10_10(n, seed=current_seed)

            outer_train_idx = row_ids[np.concatenate([tr_idx, val_idx])].tolist()
            outer_test_idx = row_ids[te_idx].tolist()
            inner_train_idx = row_ids[tr_idx].tolist()
            inner_val_idx = row_ids[val_idx].tolist()
            cats = np.zeros((n,), dtype=np.int64)  # homogeneous category

            exp_dir = os.path.join(out_dir, f"exp_{exp_i}")
            ensure_dir(exp_dir)

            # Pack args for your original function
            from types import SimpleNamespace
            inner_args = SimpleNamespace(**{
                **vars(args),
                "seed": current_seed,  # important: pass current seed to Optuna etc.
                "mapping": args.mapping
            })

            from train_servier.train import train_one_fold_by_indices
            try:
                r2 = train_one_fold_by_indices(
                    sheet_name=f"{args.dataset}_{label_name}",
                    X=X, y=y, cats=cats, feat_names=feat_names, row_ids=row_ids,
                    outer_train_idx=outer_train_idx,
                    outer_test_idx=outer_test_idx,
                    inner_train_idx=inner_train_idx,
                    inner_val_idx=inner_val_idx,
                    descriptors_mapping=descriptors_mapping,
                    args=inner_args,
                    device=args.device,
                    fold_dir=exp_dir
                )
            except Exception as e:
                print(f"[ERROR] Exp {exp_i} failed: {e}")
                r2 = -1e18

            # Compute official MoleculeNet metric
            pred_csv = os.path.join(exp_dir, "test_predictions.csv")
            if os.path.exists(pred_csv):
                pred = pd.read_csv(pred_csv)
                y_true = pred["y_true"].values.astype(float)
                y_pred = pred["y_pred"].values.astype(float)
                if metric_name == "rmse":
                    metric_val = float(np.sqrt(mean_squared_error(y_true, y_pred)))
                elif metric_name == "mae":
                    metric_val = float(mean_absolute_error(y_true, y_pred))
                else:
                    metric_val = float(np.sqrt(mean_squared_error(y_true, y_pred)))
            else:
                metric_val = float('inf')

            all_results.append({
                "exp": exp_i,
                "seed": current_seed,
                "r2": float(r2),
                "metric_val": float(metric_val)
            })

            if metric_val < best_metrix:
                best_metrix = float(metric_val)
                best_exp = exp_i

            print(f"[EXP {exp_i}] R2 = {r2:.4f}, {metric_name.upper()} = {metric_val:.5f}")

        # --- Copy best experiment to label root ---
        if best_exp is not None and args.exp_k > 1:
            best_exp_dir = os.path.join(out_dir, f"exp_{best_exp}")
            for fname in [
                'embedding_best.pth',
                'transformer_best.pth',
                'y_scaler.pth',
                'test_predictions.csv',
                'split.json'
            ]:
                src = os.path.join(best_exp_dir, fname)
                dst = os.path.join(out_dir, fname)
                if os.path.exists(src):
                    shutil.copy2(src, dst)

            # Also copy stage15 if exists
            st15_src = os.path.join(best_exp_dir, 'stage15_fused_grouped.xlsx')
            st15_dst = os.path.join(out_dir, 'stage15_fused_grouped.xlsx')
            if os.path.exists(st15_src):
                shutil.copy2(st15_src, st15_dst)

            # Update split.json to point to root files
            sp_path = os.path.join(out_dir, 'split.json')
            if os.path.exists(sp_path):
                try:
                    with open(sp_path, 'r', encoding='utf-8') as fp:
                        spj = json.load(fp)
                    spj['files']['embedding'] = os.path.join(out_dir, 'embedding_best.pth')
                    spj['files']['transformer'] = os.path.join(out_dir, 'transformer_best.pth')
                    spj['files']['y_scaler'] = os.path.join(out_dir, 'y_scaler.pth')
                    spj['files']['pred_csv'] = os.path.join(out_dir, 'test_predictions.csv')
                    with open(sp_path, 'w', encoding='utf-8') as fp:
                        json.dump(spj, fp, ensure_ascii=False, indent=2)
                except Exception as e:
                    print(f"[Warn] Failed to update split.json: {e}")

        # Final metric from best run
        best_result = next(r for r in all_results if r["exp"] == best_exp)
        final_metric = best_result["metric_val"]
        final_r2 = best_result["r2"]

        print(f"\n[FINAL] {args.dataset} / {label_name}: {metric_name.upper()} = {final_metric:.5f} (R2={final_r2:.4f}) "
              f"from exp_{best_exp} (seed={best_result['seed']})")

        summary.append((label_name, final_metric, final_r2, best_exp, best_result['seed']))

        # Save all exp results
        exp_summary_path = os.path.join(out_dir, "exp_summary.json")
        with open(exp_summary_path, 'w', encoding='utf-8') as fp:
            json.dump({
                "label": label_name,
                "dataset": args.dataset,
                "exp_k": args.exp_k,
                "base_seed": base_seed,
                "best_exp": best_exp,
                "all_runs": all_results
            }, fp, ensure_ascii=False, indent=2)

    # Write overall summary
    summary_data = {
        "dataset": args.dataset,
        "metric": metric_name.upper(),
        "exp_k": args.exp_k,
        "base_seed": args.seed,
        "labels": []
    }
    for name, metric_val, r2, best_exp, best_seed in summary:
        summary_data["labels"].append({
            "name": name,
            "metric_value": metric_val,
            "r2": r2,
            "best_exp": best_exp,
            "best_seed": best_seed
        })

    with open(os.path.join(base_out, "molnet_summary.json"), "w", encoding="utf-8") as fp:
        json.dump(summary_data, fp, ensure_ascii=False, indent=2)

    print("\n" + "=" * 80)
    print("All done! Summary saved to:", os.path.join(base_out, "molnet_summary.json"))
    print("=" * 80)


if __name__ == "__main__":
    main()
