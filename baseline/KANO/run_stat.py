
import sys
_ORIG_ARGV = sys.argv[:]
sys.argv = [sys.argv[0]]



import glob
import warnings

import pandas as pd

warnings.filterwarnings('ignore')
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

import os
import numpy as np
from other_baseline.KANO.chemprop_local.train.run_training import run_training
from other_baseline.KANO.chemprop_local.parsing import add_train_args, modify_train_args



def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def main():


    sys.argv = _ORIG_ARGV
    import argparse

    ap = argparse.ArgumentParser(
        description="KANO 批跑器：遍历 --splits_root 下的 sheet_* / fold_*/split.json，按固定划分训练并用 R² 评测"
    )
    ap.add_argument("--xlsx", required=True, help="all_data.xlsx（包含多个 sheet）")
    ap.add_argument("--splits_root", required=True, help="父目录：形如 sheet_<name>/fold_*/split.json")
    ap.add_argument("--out_root", default="./kano_runs", help="输出根目录")
    ap.add_argument("--sheets", nargs="*", help="只跑这些 sheet（默认全部）")

    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--gpu", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--exp_name_base", default="kano_batch")  # 自动注入到训练参数
    args = ap.parse_args()


    xl = pd.ExcelFile(args.xlsx)
    target_sheets = args.sheets or xl.sheet_names

    all_rows = []

    for sheet in target_sheets:
        sheet_dir = os.path.join(args.splits_root, f"sheet_{sheet}")
        split_paths = sorted(glob.glob(os.path.join(sheet_dir, "fold_*", "split.json")))
        if not split_paths:
            raise FileNotFoundError(f"[{sheet}] 未在 {sheet_dir} 下找到 fold_*/split.json")

        for sp in split_paths:
            fold_name = os.path.basename(os.path.dirname(sp))   # e.g., fold_3
            try:
                fid = int(fold_name.split('_')[-1])
            except Exception:
                fid = 1

            run_dir = os.path.join(args.out_root, f"sheet_{sheet}", fold_name)
            ensure_dir(run_dir)


            os.environ['KANO_SHEET'] = str(sheet)
            os.environ['KANO_SPLIT_JSON'] = os.path.abspath(sp)
            os.environ['KANO_FOLD'] = str(fid)


            train_argv = [
                '--data_path', os.path.abspath(args.xlsx),
                '--save_dir', os.path.abspath(run_dir),
                '--dataset_type', 'regression',
                '--metric', 'r2',
                '--epochs', str(args.epochs),
                '--batch_size', str(args.batch_size),
                '--seed', str(args.seed),
                '--exp_name', f"{args.exp_name_base}_{sheet}",
                '--exp_id', f"fold_{fid}",
                '--step', 'functional_prompt',
            ]
            if args.gpu is not None and args.gpu >= 0:
                train_argv += ['--gpu', str(args.gpu)]

            import argparse
            parser = argparse.ArgumentParser()
            add_train_args(parser)
            train_args = parser.parse_args(train_argv)
            modify_train_args(train_args)


            scores = run_training(train_args, prompt=False, logger=None)
            r2 = float(np.nanmean(scores))
            print(f"[{sheet}/{fold_name}] Test R2 = {r2:.4f}")


            test_csv = os.path.join(run_dir, "test.csv")
            pred_csv = os.path.join(run_dir, "test_preds.csv")
            if os.path.isfile(test_csv) and os.path.isfile(pred_csv):
                test_df = pd.read_csv(test_csv)
                pred_df = pd.read_csv(pred_csv)

                pred_cols = [c for c in pred_df.columns if c not in ('smiles',)]
                if len(pred_cols) >= 1:
                    pred_df = pred_df[['smiles', pred_cols[0]]].rename(columns={pred_cols[0]: 'y_pred'})
                merged = test_df.merge(pred_df, on='smiles', how='left')
                merged.rename(columns={'target': 'y_true'}, inplace=True)
                merged.to_csv(os.path.join(run_dir, "test_predictions.csv"), index=False)

            all_rows.append({"sheet": sheet, "fold": fold_name, "r2": r2})

    if all_rows:
        summ = pd.DataFrame(all_rows)
        ensure_dir(args.out_root)
        summary_path = os.path.join(args.out_root, "summary.csv")
        summ.to_csv(summary_path, index=False)
        print("Summary saved:", summary_path)


if __name__ == "__main__":
    print("[INFO] Using batch runner:", os.path.abspath(__file__))
    main()