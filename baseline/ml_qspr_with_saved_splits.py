
import os, json, glob, argparse, warnings
from typing import Dict, Tuple, List
import numpy as np
import pandas as pd

from joblib import dump, Parallel, delayed
from scipy.io import savemat

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import KFold, ParameterGrid, cross_val_score
from sklearn.exceptions import ConvergenceWarning


from sklearn.linear_model import (
    Ridge, Lasso, ElasticNet, BayesianRidge, HuberRegressor,
    OrthogonalMatchingPursuit, LassoLars, SGDRegressor
)

from sklearn.neighbors import KNeighborsRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor,
    AdaBoostRegressor, BaggingRegressor
)
from sklearn.neural_network import MLPRegressor

warnings.filterwarnings("ignore", category=ConvergenceWarning)




def ensure_dir(p): os.makedirs(p, exist_ok=True)

def list_sheets_from_dir(data_dir: str) -> List[str]:
    sheets = []
    for p in sorted(glob.glob(os.path.join(data_dir, "*.csv"))):
        sheets.append(os.path.splitext(os.path.basename(p))[0])
    return sheets

def find_fold_paths(root, sheet):
    pats = [
        os.path.join(root, sheet, "fold_*", "split.json"),
        os.path.join(root, f"sheet_{sheet}", "fold_*", "split.json"),
    ]
    for pat in pats:
        paths = sorted(glob.glob(pat))
        if paths: return paths
    return []

def read_sheet_csv(path: str, category_col_idx=0, label_col_idx=1) -> Tuple[pd.DataFrame, np.ndarray, pd.Series, List[str]]:

    df = pd.read_csv(path)

    cat_series = df.iloc[:, category_col_idx]
    y_raw = pd.to_numeric(df.iloc[:, label_col_idx], errors="coerce").values.astype(np.float64)
    Xdf = df.iloc[:, 2:].copy()

    for c in Xdf.columns:
        Xdf[c] = pd.to_numeric(Xdf[c], errors="coerce")
    Xdf = Xdf.replace([np.inf, -np.inf], np.nan)
    feat_names = list(Xdf.columns)
    return Xdf, y_raw, cat_series, feat_names


def make_models(random_state=42):
    lin_alphas = np.logspace(-6, 1, 7)
    def scaled():
        return [
            ("imputer", SimpleImputer(strategy="median")),
            ("var", VarianceThreshold(0.0)),
            ("scaler", StandardScaler()),
        ]
    base_pre = [("imputer", SimpleImputer(strategy="median")), ("var", VarianceThreshold(0.0))]

    models = {

        "ridge":      (Pipeline(scaled() + [("est", Ridge(random_state=random_state))]),
                       {"est__alpha": lin_alphas}),
        # "lasso":      (Pipeline(scaled() + [("est", Lasso(max_iter=5000, random_state=random_state))]),
        #                {"est__alpha": lin_alphas}),
        # "elastic":    (Pipeline(scaled() + [("est", ElasticNet(max_iter=5000, random_state=random_state))]),
        #                {"est__alpha": lin_alphas, "est__l1_ratio":[0.1,0.3,0.5,0.7,0.9]}),
        # "bayesridge": (Pipeline(scaled() + [("est", BayesianRidge())]), {}),
        # "huber":      (Pipeline(scaled() + [("est", HuberRegressor(epsilon=1.35, max_iter=1000))]), {}),
        # "omp":        (Pipeline(scaled() + [("est", OrthogonalMatchingPursuit())]),
        #                {"est__n_nonzero_coefs":[None,5,10,20]}),
        # "lassolars":  (Pipeline(scaled() + [("est", LassoLars(max_iter=1000))]),
        #                {"est__alpha": lin_alphas}),
        "sgd":        (Pipeline(scaled() + [("est", SGDRegressor(loss='huber', max_iter=5000, random_state=random_state))]),
                       {"est__alpha":[1e-6,1e-5,1e-4], "est__eta0":[1e-3,1e-2]}),


        # "svr_rbf": (Pipeline(scaled() + [("est", SVR(kernel="rbf"))]),
        #             {"est__C":[0.1,1,10,100], "est__epsilon":[0.01,0.1,0.5], "est__gamma":["scale","auto"]}),
        # "svr_lin": (Pipeline(scaled() + [("est", SVR(kernel="linear"))]),
        #             {"est__C":[0.1,1,10,100], "est__epsilon":[0.01,0.1,0.5]}),
        # "krr_rbf": (Pipeline(scaled() + [("est", KernelRidge(kernel="rbf"))]),
        #             {"est__alpha":[1e-3,1e-2,1e-1,1.0], "est__gamma":[1e-3,1e-2,1e-1,1.0]}),
        "knn":     (Pipeline(scaled() + [("est", KNeighborsRegressor())]),
                    {"est__n_neighbors":[3,5,7,11], "est__weights":["uniform","distance"]}),


        # "dt":  (Pipeline(base_pre + [("est", DecisionTreeRegressor(random_state=random_state))]),
        #         {"est__max_depth":[None,5,10,20], "est__min_samples_leaf":[1,2,4]}),
        "rf":  (Pipeline(base_pre + [("est", RandomForestRegressor(n_estimators=800, n_jobs=-1, random_state=random_state))]),
                {"est__max_depth":[None,10,20,30], "est__min_samples_leaf":[1,2,4]}),
        # "et":  (Pipeline(base_pre + [("est", ExtraTreesRegressor(n_estimators=1000, n_jobs=-1, random_state=random_state))]),
        #         {"est__max_depth":[None,10,20,30], "est__min_samples_leaf":[1,2,4]}),
        # "gbr": (Pipeline(base_pre + [("est", GradientBoostingRegressor(random_state=random_state, n_estimators=1200, subsample=0.8))]),
        #         {"est__learning_rate":[0.05,0.1,0.2], "est__max_depth":[2,3,4]}),
        # "ada": (Pipeline(base_pre + [("est", AdaBoostRegressor(random_state=random_state, n_estimators=800))]),
        #         {"est__learning_rate":[0.05,0.1,0.2], "est__loss":["linear","square","exponential"]}),
        # "bag": (Pipeline(base_pre + [("est", BaggingRegressor(random_state=random_state, n_estimators=400, n_jobs=-1))]),
        #         {"est__max_features":[0.7,0.9,1.0]}),

        # MLP
        # "mlp": (Pipeline(scaled() + [("est", MLPRegressor(max_iter=2500, random_state=random_state))]),
        #         {"est__hidden_layer_sizes":[(64,),(128,),(64,64)], "est__alpha":[1e-5,1e-4,1e-3]}),
    }
    return models


def cv_rmse(pipe: Pipeline, X: np.ndarray, y: np.ndarray, cv, n_jobs=1) -> Tuple[float, float]:
    scores = cross_val_score(pipe, X, y, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=n_jobs)
    rmses = -scores
    return float(np.mean(rmses)), float(np.std(rmses))

def select_best_by_cv(name, mdef, X, y, tr_idx, cv_splits: int, seed: int, n_jobs: int):
    X_tr, y_tr = X[tr_idx], y[tr_idx]
    kf = KFold(n_splits=cv_splits, shuffle=True, random_state=seed)
    pipe, grid = mdef
    best = {"rmse": 1e18, "rmse_std": 1e18, "params": None, "model": None}
    for params in ParameterGrid(grid) if grid else [dict()]:
        mdl = Pipeline(pipe.steps, verbose=False)
        if params: mdl.set_params(**params)
        rmse_mean, rmse_std = cv_rmse(mdl, X_tr, y_tr, kf, n_jobs=n_jobs)
        if (rmse_mean < best["rmse"]) or (abs(rmse_mean - best["rmse"]) < 1e-12 and rmse_std < best["rmse_std"]):
            best = {"rmse": rmse_mean, "rmse_std": rmse_std, "params": params, "model": mdl}
    best["model"].set_params(**(best["params"] or {}))
    best["model"].fit(X_tr, y_tr)  # 便于后续复用
    return name, best

# ----------------- 主流程 -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--splits_root", required=True)
    ap.add_argument("--out_root", default="./ml_qspr_uob3_runs_from_csv")
    ap.add_argument("--sheets", nargs="*")
    ap.add_argument("--inner_cv", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_jobs", type=int, default=-1)
    args = ap.parse_args()

    ensure_dir(args.out_root)
    models = make_models(random_state=args.seed)


    sheets = args.sheets or list_sheets_from_dir(args.data_dir)
    if not sheets:
        raise FileNotFoundError(f"No CSV files were found in {args.data_dir}.")

    summary_rows = []

    for sheet in sheets:
        print(f"\n===== Sheet: {sheet} =====")
        csv_path = os.path.join(args.data_dir, f"{sheet}.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"{csv_path} not exist.")
        Xdf, y_raw, cat_series, feat_names = read_sheet_csv(csv_path)


        Xdf = Xdf.reset_index(drop=True)
        y_raw = y_raw.reshape(-1)
        X = Xdf.values.astype(np.float64)
        n_total = X.shape[0]
        print(f"[{sheet}] matrix shape: {X.shape} (rows=samples, cols=features={len(feat_names)})")

        # splits
        fold_paths = find_fold_paths(args.splits_root, sheet)
        if not fold_paths:
            raise FileNotFoundError(f"[{sheet}] not find split.json：{args.splits_root}/(sheet_){sheet}/fold_x/split.json")

        sheet_dir = os.path.join(args.out_root, f"sheet_{sheet}")
        ensure_dir(sheet_dir)

        for spath in fold_paths:
            fold_id = os.path.basename(os.path.dirname(spath))
            run_dir = os.path.join(sheet_dir, fold_id)
            ensure_dir(run_dir)

            with open(spath, "r", encoding="utf-8") as f:
                sp = json.load(f)
            tr_idx, va_idx, te_idx = sp["inner_train_idx"], sp["inner_val_idx"], sp["outer_test_idx"]

            for name, idxs in [("train", tr_idx), ("val", va_idx), ("test", te_idx)]:
                if len(idxs) == 0: raise ValueError(f"[{sheet}/{fold_id}] {name} split is empty.")

            print(f"[{sheet}/{fold_id}] sizes: train={len(tr_idx)}, val={len(va_idx)}, test={len(te_idx)}")


            const_r2 = float(r2_score(y_raw[te_idx], np.full_like(y_raw[te_idx], np.mean(y_raw[tr_idx]))))
            print(f"[{sheet}/{fold_id}] Constant baseline Test R2 = {const_r2:.4f}")



            results = Parallel(n_jobs=args.n_jobs, verbose=0)(
                delayed(select_by_cv_wrapper)(name, mdef, X, y_raw, tr_idx, args.inner_cv, args.seed, args.n_jobs)
                for name, mdef in models.items()
            )

            best_name, best_info = min(results, key=lambda x: (x[1]["rmse"], x[1]["rmse_std"]))
            print(f"[{sheet}/{fold_id}] Best by inner-CV RMSE: {best_name} "
                  f"(mean={best_info['rmse']:.6g}, std={best_info['rmse_std']:.6g}, params={best_info['params']})")


            val_model = Pipeline(best_info["model"].steps, verbose=False)
            if best_info["params"]: val_model.set_params(**best_info["params"])
            val_model.fit(X[tr_idx], y_raw[tr_idx])
            y_val_pred = val_model.predict(X[va_idx])
            val_rmse = float(np.sqrt(mean_squared_error(y_raw[va_idx], y_val_pred)))
            val_r2   = float(r2_score(y_raw[va_idx], y_val_pred))
            print(f"[{sheet}/{fold_id}] Inner-val RMSE={val_rmse:.6g}, R2={val_r2:.4f}")


            idx_tv = np.array(sorted(tr_idx + va_idx))
            final_model = Pipeline(best_info["model"].steps, verbose=False)
            if best_info["params"]: final_model.set_params(**best_info["params"])
            final_model.fit(X[idx_tv], y_raw[idx_tv])


            y_pred = final_model.predict(X[te_idx]).reshape(-1)
            y_true = y_raw[te_idx]
            test_rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
            test_r2   = float(r2_score(y_true, y_pred))
            print(f"[{sheet}/{fold_id}] Test RMSE={test_rmse:.6g}  R2={test_r2:.4f}")


            out_df = pd.DataFrame({"row_id": te_idx, "y_true": y_true, "y_pred": y_pred})
            out_df.insert(1, "category", cat_series.iloc[te_idx].values)
            out_df = out_df.sort_values("row_id")
            out_df.to_csv(os.path.join(run_dir, "test_predictions.csv"), index=False)


            val_scores_json = {name: {"cv_rmse_mean": info["rmse"], "cv_rmse_std": info["rmse_std"]}
                               for name, info in results}
            with open(os.path.join(run_dir, "fold_metrics.json"), "w", encoding="utf-8") as f:
                json.dump({
                    "sheet": sheet, "fold": fold_id,
                    "best_model": best_name,
                    "best_params": best_info["params"],
                    "inner_cv": args.inner_cv,
                    "val_sanity_rmse": val_rmse, "val_sanity_r2": val_r2,
                    "test_rmse": test_rmse, "test_r2": test_r2,
                    "feature_names": feat_names
                }, f, ensure_ascii=False, indent=2)


            dump(final_model, os.path.join(run_dir, "best_model.joblib"))


            mat_dict = {
                "trainedModel_name": np.array(best_name),
                "best_params_json": np.array(json.dumps(best_info["params"] or {})),
                "Matrix_train": X[idx_tv].astype(np.float64),
                "y_train": y_raw[idx_tv].astype(np.float64),
                "feature_names": np.array(feat_names, dtype=object),
                "note": np.array("Variable formats and feature order must match the training Matrix."),
            }

            try:
                est = final_model.named_steps["est"]
                if hasattr(est, "coef_"):
                    mat_dict["coef_"] = np.atleast_2d(est.coef_).astype(np.float64)
                if hasattr(est, "intercept_"):
                    mat_dict["intercept_"] = np.array([est.intercept_], dtype=np.float64)
            except Exception:
                pass
            savemat(os.path.join(run_dir, "best_model.mat"), mat_dict)


            summary_rows.append({"sheet": sheet, "fold": fold_id, "r2": test_r2, "rmse": test_rmse})


    if summary_rows:
        summ = pd.DataFrame(summary_rows)
        summ.to_csv(os.path.join(args.out_root, "summary.csv"), index=False)
        agg = summ.groupby("sheet").agg(mean_r2=("r2","mean"), std_r2=("r2","std"),
                                        mean_rmse=("rmse","mean"), std_rmse=("rmse","std"),
                                        n=("r2","count")).reset_index()
        agg.to_csv(os.path.join(args.out_root, "summary_agg.csv"), index=False)
        print("\nSaved:", os.path.join(args.out_root, "summary.csv"), "and summary_agg.csv")
    else:
        print("No results collected.")


def select_by_cv_wrapper(name, mdef, X, y, tr_idx, inner_cv, seed, n_jobs):
    return select_best_by_cv(name, mdef, X, y, tr_idx, inner_cv, seed, n_jobs)

if __name__ == "__main__":
    main()
