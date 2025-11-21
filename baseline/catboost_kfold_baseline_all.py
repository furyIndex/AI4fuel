
import sys
_ORIG_ARGV = sys.argv[:]
sys.argv = [sys.argv[0]]

import argparse
import json
import os
import random
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
import optuna
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# -------------------- Utils --------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)


def class_counts(cats: np.ndarray) -> Dict:
    counts = {}
    for key in cats:
        counts[key] = counts.get(key, 0) + 1
    return counts


def build_stratified_folds_by_category(cats: np.ndarray, n_splits: int, seed: int):

    rng = np.random.RandomState(seed)

    by_cls = {}
    for i, key in enumerate(cats):
        by_cls.setdefault(key, []).append(i)


    cls_chunks = {}
    for key, idxs_list in by_cls.items():
        idxs = np.array(idxs_list, dtype=int)
        rng.shuffle(idxs)
        L = len(idxs)
        chunks = []
        start = 0
        for s in range(n_splits):
            size = L // n_splits + (1 if s < (L % n_splits) else 0)
            end = start + size
            chunk = idxs[start:end]
            chunks.append(chunk)
            start = end
        cls_chunks[key] = chunks


    folds = []
    for s in range(n_splits):
        test_list = [cls_chunks[key][s] for key in cls_chunks]
        test_idx = np.concatenate(test_list) if len(test_list) > 1 else (test_list[0] if test_list else np.array([], dtype=int))
        mask = np.ones(len(cats), dtype=bool)
        mask[test_idx] = False
        trainval_idx = np.where(mask)[0]
        folds.append((trainval_idx, test_idx))
    return folds


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


def get_catboost_model(args, **best_params):

    common = dict(
        iterations=best_params.get('iterations', args.iterations[-1]),
        learning_rate=best_params.get('learning_rate', args.lrs[-1]),
        depth=best_params.get('depth', args.depths[-1]),
        l2_leaf_reg=best_params.get('l2_leaf_reg', args.l2_regs[-1]),
        loss_function='RMSE',
        eval_metric=('RMSE' if args.gpu else 'R2'),
        random_seed=args.seed,
        od_type='Iter',
        od_wait=args.patience,
        use_best_model=True,
        verbose=False,
        thread_count=-1,
    )
    if args.gpu:
        common['task_type'] = 'GPU'
        if args.gpu_devices is not None:
            common['devices'] = args.gpu_devices
    else:
        common['task_type'] = 'CPU'
    return CatBoostRegressor(**common)


def run_one_sheet_with_split_dir(args, xlsx_path: str, sheet: str, split_dir_for_sheet: str):

    X, y, cats, feat_names, row_ids = load_sheet(xlsx_path, sheet)
    base_root = os.path.join(args.save_dir, f"sheet_{sheet}")
    run_root = os.path.join(base_root, "catboost")
    ensure_dir(run_root)

    cv_manifest = {
        "sheet": sheet,
        "seed": int(args.seed),
        "stratified_by": "category",
        "save_dir": run_root,
        "model": "CatBoostRegressor",
        "folds": []
    }


    fold_splits = []
    for i in range(1, 6):
        sp_path = os.path.join(split_dir_for_sheet, f"fold_{i}", "split.json")
        if not os.path.exists(sp_path):
            raise FileNotFoundError(f"Can not find split.json: {sp_path}")
        inner_train_idx, inner_val_idx, outer_test_idx = read_split_json(sp_path)
        fold_splits.append((inner_train_idx, inner_val_idx, outer_test_idx))

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

        def objective(trial):
            iterations = trial.suggest_categorical('iterations', args.iterations)
            depth = trial.suggest_categorical('depth', args.depths)
            learning_rate = trial.suggest_categorical('learning_rate', args.lrs)
            l2_leaf_reg = trial.suggest_categorical('l2_leaf_reg', args.l2_regs)

            model = get_catboost_model(
                args,
                iterations=iterations,
                depth=depth,
                learning_rate=learning_rate,
                l2_leaf_reg=l2_leaf_reg
            )
            model.fit(X_tr, y_tr, eval_set=(X_val, y_val))
            yhat_val = model.predict(X_val)
            r2 = float(r2_score(y_val, yhat_val))
            return r2

        study = optuna.create_study(direction='maximize', study_name=f"{sheet}_catboost_fold_{fold_id}")
        study.optimize(objective, n_trials=args.trials, show_progress_bar=False)

        best_params = study.best_trial.params

        best_model = get_catboost_model(args, **best_params)
        best_model.fit(X_tr, y_tr, eval_set=(X_val, y_val))

        yhat_te = best_model.predict(X_te)
        r2 = float(r2_score(y_te, yhat_te))
        mae = float(mean_absolute_error(y_te, yhat_te))
        mse = float(mean_squared_error(y_te, yhat_te))
        fold_metrics.append(r2)
        print(f"[{sheet}] Fold {fold_id}: Test R2 = {r2:.4f}")

        model_path = os.path.join(fold_dir, 'catboost_best.cbm')
        best_model.save_model(model_path)

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
            "stratified_by": "category",
            "per_class_all": per_class_all,
            "per_class_outer_test": per_class_outer_test,
            "outer_train_idx": [int(x) for x in np.setdiff1d(np.arange(len(X)), outer_test_idx)],
            "outer_test_idx": [int(x) for x in outer_test_idx],
            "inner_train_idx": [int(x) for x in inner_train_idx],
            "inner_val_idx": [int(x) for x in inner_val_idx],
            "best_params": best_params,
            "metrics": {"test_r2": r2, "test_mae": mae, "test_mse": mse},
            "files": {
                "catboost": model_path,
                "pred_csv": os.path.join(fold_dir, 'test_predictions.csv')
            }
        }
        with open(os.path.join(fold_dir, 'split.json'), 'w', encoding='utf-8') as fp:
            json.dump(split_info, fp, ensure_ascii=False, indent=2)

        cv_manifest["folds"].append({
            "fold_id": int(fold_id),
            "test_r2": r2,
            "split_file": os.path.join(fold_dir, 'split.json'),
            "catboost": model_path,
            "pred_csv": os.path.join(fold_dir, 'test_predictions.csv')
        })

    with open(os.path.join(run_root, 'cv_manifest.json'), 'w', encoding='utf-8') as fp:
        json.dump(cv_manifest, fp, ensure_ascii=False, indent=2)

    return float(np.mean(fold_metrics)), float(np.std(fold_metrics)), [round(v, 4) for v in fold_metrics]


def main():
    sys.argv = _ORIG_ARGV
    ap = argparse.ArgumentParser()
    ap.add_argument('--xlsx', required=True, help='An Excel file containing all the descriptors')
    ap.add_argument('--sheet', help='Train only this sheet')
    ap.add_argument('--save_dir', default='./baseline_runs')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--trials', type=int, default=20)
    ap.add_argument('--patience', type=int, default=50)


    ap.add_argument('--iterations', nargs='+', type=int, default=[500, 800, 1000, 1200, 1500])
    ap.add_argument('--depths', nargs='+', type=int, default=[2, 4, 6])
    ap.add_argument('--lrs', nargs='+', type=float, default=[0.01, 0.02, 0.03, 0.05])
    ap.add_argument('--l2_regs', nargs='+', type=float, default=[1.0, 3.0, 5.0, 10.0])


    ap.add_argument('--gpu', action='store_true')
    ap.add_argument('--gpu_devices', default='0')


    ap.add_argument('--split_dir', type=str, default=None,
                    help='The directory where the corresponding partition is located (in the form of save_dir/sheet_<sheet>)')
    ap.add_argument('--split_root', type=str, default=None,
                    help='Batch mode: The root directory contains multiple subdirectories named "sheet_*", and within each of these directories, there is a file named "fold_i/split.json".')

    args = ap.parse_args()
    set_seed(args.seed)


    if args.split_root:

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
            mean_r2, std_r2, r2_list = run_one_sheet_with_split_dir(args, args.xlsx, sheet, split_dir_for_sheet)
            summary.append((sheet, mean_r2, std_r2, r2_list))


        print("=" * 70)
        print("baseline completed（CatBoost）")
        for sheet, mean_r2, std_r2, r2_list in summary:
            print(f"{sheet:>24}: R2={mean_r2:.4f} ± {std_r2:.4f} | folds={r2_list}")
        return

    if not args.sheet:
        raise ValueError("No --sheet or --split_root was provided; please choose one of them.")

    if args.split_dir:

        run_one_sheet_with_split_dir(args, args.xlsx, args.sheet, args.split_dir)
    else:

        X, y, cats, feat_names, row_ids = load_sheet(args.xlsx, args.sheet)
        base_root = os.path.join(args.save_dir, f"sheet_{args.sheet}")
        run_root = os.path.join(base_root, "catboost")
        ensure_dir(run_root)

        folds = build_stratified_folds_by_category(cats, n_splits=5, seed=args.seed)
        fold_splits = []
        for trval_idx, te_idx in folds:
            rel_ids = np.arange(len(trval_idx))
            c_trval = cats[trval_idx]
            try:
                rel_tr, rel_val = train_test_split(
                    rel_ids, test_size=0.2, random_state=args.seed, shuffle=True, stratify=c_trval
                )
            except Exception:
                rel_tr, rel_val = train_test_split(
                    rel_ids, test_size=0.2, random_state=args.seed, shuffle=True, stratify=None
                )
            inner_train_idx = row_ids[trval_idx[rel_tr]]
            inner_val_idx = row_ids[trval_idx[rel_val]]
            outer_test_idx = row_ids[te_idx]
            fold_splits.append((inner_train_idx, inner_val_idx, outer_test_idx))

        cv_manifest = {
            "sheet": args.sheet,
            "seed": int(args.seed),
            "stratified_by": "category",
            "save_dir": run_root,
            "model": "CatBoostRegressor",
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

            def objective(trial):
                iterations = trial.suggest_categorical('iterations', args.iterations)
                depth = trial.suggest_categorical('depth', args.depths)
                learning_rate = trial.suggest_categorical('learning_rate', args.lrs)
                l2_leaf_reg = trial.suggest_categorical('l2_leaf_reg', args.l2_regs)

                model = get_catboost_model(
                    args,
                    iterations=iterations,
                    depth=depth,
                    learning_rate=learning_rate,
                    l2_leaf_reg=l2_leaf_reg
                )
                model.fit(X_tr, y_tr, eval_set=(X_val, y_val))
                yhat_val = model.predict(X_val)
                r2 = float(r2_score(y_val, yhat_val))
                return r2

            study = optuna.create_study(direction='maximize', study_name=f"{args.sheet}_catboost_fold_{fold_id}")
            study.optimize(objective, n_trials=args.trials, show_progress_bar=False)

            best_params = study.best_trial.params
            best_model = get_catboost_model(args, **best_params)
            best_model.fit(X_tr, y_tr, eval_set=(X_val, y_val))

            yhat_te = best_model.predict(X_te)
            r2 = float(r2_score(y_te, yhat_te))
            mae = float(mean_absolute_error(y_te, yhat_te))
            mse = float(mean_squared_error(y_te, yhat_te))
            fold_metrics.append(r2)
            print(f"Fold {fold_id}: Test R2 = {r2:.4f}")

            model_path = os.path.join(fold_dir, 'catboost_best.cbm')
            best_model.save_model(model_path)

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
                "sheet": args.sheet,
                "seed": int(args.seed),
                "stratified_by": "category",
                "per_class_all": per_class_all,
                "per_class_outer_test": per_class_outer_test,
                "outer_train_idx": [int(x) for x in np.setdiff1d(np.arange(len(X)), outer_test_idx)],
                "outer_test_idx": [int(x) for x in outer_test_idx],
                "inner_train_idx": [int(x) for x in inner_train_idx],
                "inner_val_idx": [int(x) for x in inner_val_idx],
                "best_params": best_params,
                "metrics": {"test_r2": r2, "test_mae": mae, "test_mse": mse},
                "files": {
                    "catboost": model_path,
                    "pred_csv": os.path.join(fold_dir, 'test_predictions.csv')
                }
            }
            with open(os.path.join(fold_dir, 'split.json'), 'w', encoding='utf-8') as fp:
                json.dump(split_info, fp, ensure_ascii=False, indent=2)

            cv_manifest["folds"].append({
                "fold_id": int(fold_id),
                "test_r2": r2,
                "split_file": os.path.join(fold_dir, 'split.json'),
                "catboost": model_path,
                "pred_csv": os.path.join(fold_dir, 'test_predictions.csv')
            })

        with open(os.path.join(run_root, 'cv_manifest.json'), 'w', encoding='utf-8') as fp:
            json.dump(cv_manifest, fp, ensure_ascii=False, indent=2)

        r_list = [round(r, 4) for r in fold_metrics]
        print("=" * 60)
        print("5-fold R2s:", r_list)
        print("Mean R2:", float(np.mean(fold_metrics)))
        print("Std  R2:", float(np.std(fold_metrics)))


if __name__ == '__main__':
    main()
