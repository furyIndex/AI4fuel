
import argparse, os, json, glob, subprocess
import pandas as pd

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def read_excel_sheet(xlsx_path, sheet):
    return pd.read_excel(xlsx_path, sheet_name=sheet)

def find_col(df, preferred, fallback_second_last=False):
    m = {c.lower(): c for c in df.columns}
    if preferred and preferred.lower() in m:
        return m[preferred.lower()]
    if fallback_second_last:
        return df.columns[-2]
    return preferred

def load_split(path):
    with open(path, "r", encoding="utf-8") as f:
        sp = json.load(f)
    return sp["inner_train_idx"], sp["inner_val_idx"], sp["outer_test_idx"]

def run_train(train_csv, val_csv, test_csv, out_dir,
              smiles_col, label_col, seed, epochs, gpu, extra_args=None):
    args = [
        "chemprop_train",
        "--data_path", train_csv,
        "--separate_val_path", val_csv,
        "--separate_test_path", test_csv,
        "--dataset_type", "regression",
        "--metric", "r2",
        "--save_dir", out_dir,
        "--smiles_column", smiles_col,
        "--target_columns", label_col,
        "--epochs", str(epochs),
        "--seed", str(seed),
        "--num_folds", "1",
        "--ensemble_size", "1"
    ]
    if gpu is not None and gpu >= 0:
        args += ["--gpu", str(gpu)]
    if extra_args:
        args += extra_args
    print(">>", " ".join(args))
    subprocess.run(args, check=True)

def run_predict(test_csv, ckpt_dir, preds_csv, smiles_col, gpu):
    args = [
        "chemprop_predict",
        "--test_path", test_csv,
        "--checkpoint_dir", ckpt_dir,
        "--preds_path", preds_csv,
        "--smiles_column", smiles_col,
    ]
    if gpu is not None and gpu >= 0:
        args += ["--gpu", str(gpu)]
    print(">>", " ".join(args))
    subprocess.run(args, check=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", required=True)
    ap.add_argument("--splits_root", required=True)
    ap.add_argument("--out_root", default="./chemprop_v1_runs")
    ap.add_argument("--smiles_col", default="SMILES")
    ap.add_argument("--label_col", default="label")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gpu", type=int, default=-1)
    ap.add_argument("--sheets", nargs="*")
    args = ap.parse_args()

    xls = pd.ExcelFile(args.xlsx)
    sheets = args.sheets or xls.sheet_names

    os.makedirs(args.out_root, exist_ok=True)
    all_rows = []

    for sheet in sheets:
        print(f"\n===== Sheet: {sheet} =====")
        df = read_excel_sheet(args.xlsx, sheet)
        smiles_col = find_col(df, args.smiles_col)
        label_col = find_col(df, args.label_col, fallback_second_last=True)


        category_col = df.columns[0] if df.columns[0] not in (smiles_col, label_col) else None


        base_df = pd.DataFrame({smiles_col: df[smiles_col], label_col: df[label_col]})
        if category_col is not None:
            base_df[category_col] = df[category_col]

        sheet_dir = os.path.join(args.out_root, f"sheet_{sheet}")
        os.makedirs(sheet_dir, exist_ok=True)

        fold_paths = sorted(glob.glob(os.path.join(args.splits_root, f"sheet_{sheet}", "fold_*", "split.json")))
        if not fold_paths:
            raise FileNotFoundError(f"[The file fold_x/split.json was not found in {os.path.join(args.splits_root, sheet)}.")

        for split_path in fold_paths:
            print("=========================={}============================".format(split_path))
            fold_id = os.path.basename(os.path.dirname(split_path))
            run_dir = os.path.join(sheet_dir, fold_id)
            os.makedirs(run_dir, exist_ok=True)

            tr_idx, va_idx, te_idx = load_split(split_path)
            train_df = base_df.iloc[tr_idx].copy()
            val_df   = base_df.iloc[va_idx].copy()
            test_df  = base_df.iloc[te_idx].copy()

            train_csv = os.path.join(run_dir, "train.csv")
            val_csv   = os.path.join(run_dir, "val.csv")
            test_csv  = os.path.join(run_dir, "test.csv")
            train_df[[smiles_col, label_col]].to_csv(train_csv, index=False)
            val_df[[smiles_col, label_col]].to_csv(val_csv, index=False)
            test_df[[smiles_col, label_col]].to_csv(test_csv, index=False)

            out_train = os.path.join(run_dir, "train_outputs")
            os.makedirs(out_train, exist_ok=True)


            run_train(train_csv, val_csv, test_csv, out_train,
                      smiles_col, label_col, args.seed, args.epochs, args.gpu)


            preds_csv = os.path.join(run_dir, "test_preds.csv")
            run_predict(test_csv, out_train, preds_csv, smiles_col, args.gpu)


            preds = pd.read_csv(preds_csv)
            pred_col = [c for c in preds.columns if c != smiles_col][0]
            merged = test_df.reset_index().rename(columns={"index": "row_id"})
            merged["y_true"] = merged[label_col].values
            merged["y_pred"] = preds[pred_col].values
            cols = ["row_id"]
            if category_col is not None:
                merged.rename(columns={category_col: "category"}, inplace=True)
                cols.append("category")
            cols += ["y_true", "y_pred"]
            merged[cols].to_csv(os.path.join(run_dir, "test_predictions.csv"), index=False)


            from sklearn.metrics import r2_score
            r2 = float(r2_score(merged["y_true"], merged["y_pred"]))
            all_rows.append({"sheet": sheet, "fold": fold_id, "r2": r2})
            print(f"[{sheet}/{fold_id}] Test R2 = {r2:.4f}")

    if all_rows:
        summ = pd.DataFrame(all_rows)
        summ.to_csv(os.path.join(args.out_root, "summary.csv"), index=False)
        print("Summary saved:", os.path.join(args.out_root, "summary.csv"))

if __name__ == "__main__":
    main()
