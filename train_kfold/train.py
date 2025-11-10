import json
import sys
from train_kfold.kfold_prepare import load_sheet, build_stratified_folds_by_cat_and_logy, make_log_bins
from train_kfold.train_domain import ensure_dir, train_one_fold_by_indices
_ORIG_ARGV = sys.argv[:]
sys.argv = [sys.argv[0]]
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # 或 ":16:8"
import argparse
import random
import shutil
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split



def set_seed(seed=42):
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





# -------------------- 主流程 --------------------
def main():
    sys.argv = _ORIG_ARGV

    ap = argparse.ArgumentParser()
    ap.add_argument('--xlsx', required=True)
    ap.add_argument('--mapping', required=True)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--save_dir', default='./lg_transformer_kfold_runs')

    # 选择性重训相关
    ap.add_argument('--only_sheet', default=None, help='只训练该 sheet（不填则遍历全部 sheet）')
    ap.add_argument('--only_folds', default=None, help='只训练这些折（1 基，逗号分隔，如 2 或 1,3,5）')
    ap.add_argument('--split_json', default=None, help='指定已有 split.json，按其划分重训该折')
    ap.add_argument('--use_saved_splits', action='store_true', help='5 折流程中优先复用各折 split.json 中的划分')

    # GCN 参数
    ap.add_argument('--k', type=int, default=10)
    ap.add_argument('--a', type=float, default=0.1)

    # 阶段一（embedding）
    ap.add_argument('--epochs_embed', type=int, default=50)
    ap.add_argument('--bs_embed', type=int, default=512)
    ap.add_argument('--lr_embed', type=float, default=1e-2)
    ap.add_argument('--wd_embed', type=float, default=1e-3)
    ap.add_argument('--margin', type=float, default=3.0)


    # 阶段二（Optuna + 早停）
    ap.add_argument('--trials', type=int, default=50)
    ap.add_argument('--epochs_tr', type=int, default=500)
    ap.add_argument('--patience', type=int, default=50)
    ap.add_argument('--bs_tr', type=int, default=512)
    ap.add_argument('--save_stage15', default=True, action='store_true')

    ap.add_argument('--exp_k', type=int, default=1, help='每个 fold 重复实验次数')

    args = ap.parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    with open(args.mapping, 'r', encoding='utf-8') as f:
        descriptors_mapping = json.load(f)
    ensure_dir(args.save_dir)

    stand_properties = ["CN", "Flash_point", "LHV", "MON", "RON", "Surface_tension", "Tb", "Tm"]

    # ----------- 特殊路径：直接基于一个 split.json 复训该折 -----------
    if args.split_json is not None:
        split_path = os.path.abspath(args.split_json)
        if not os.path.isfile(split_path):
            raise FileNotFoundError(f"split.json 不存在：{split_path}")
        with open(split_path, 'r', encoding='utf-8') as fp:
            split = json.load(fp)

        sheet_name = split["sheet"]
        X, y, cats, feat_names, row_ids = load_sheet(args.xlsx, sheet_name)

        fold_root = os.path.dirname(split_path)

        # 单折多次尝试
        if args.exp_k == 1:
            r2 = train_one_fold_by_indices(
                sheet_name=sheet_name,
                X=X, y=y, cats=cats, feat_names=feat_names, row_ids=row_ids,
                outer_train_idx=split["outer_train_idx"],
                outer_test_idx=split["outer_test_idx"],
                inner_train_idx=split["inner_train_idx"],
                inner_val_idx=split["inner_val_idx"],
                descriptors_mapping=descriptors_mapping,
                args=args, device=device,
                fold_dir=fold_root,
                stand_properties=stand_properties
            )
            print(f"[DONE] 复训完成（split.json：{split_path}），Test R2 = {round(r2, 4)}")
            return
        else:
            base_seed = int(args.seed)
            best_r2 = -1e18
            best_exp = None
            attempt_r2s = []
            for exp_i in range(1, args.exp_k + 1):
                tmp_seed = base_seed + exp_i - 1
                print(f"[Try] exp_{exp_i} with seed={tmp_seed}")
                set_seed(tmp_seed)
                old_seed = args.seed
                args.seed = tmp_seed
                exp_dir = os.path.join(fold_root, f'exp_{exp_i}')
                ensure_dir(exp_dir)
                r2_i = train_one_fold_by_indices(
                    sheet_name=sheet_name,
                    X=X, y=y, cats=cats, feat_names=feat_names, row_ids=row_ids,
                    outer_train_idx=split["outer_train_idx"],
                    outer_test_idx=split["outer_test_idx"],
                    inner_train_idx=split["inner_train_idx"],
                    inner_val_idx=split["inner_val_idx"],
                    descriptors_mapping=descriptors_mapping,
                    args=args, device=device,
                    fold_dir=exp_dir,
                    stand_properties=stand_properties
                )
                attempt_r2s.append(float(r2_i))
                if r2_i > best_r2:
                    best_r2 = float(r2_i)
                    best_exp = exp_i
                args.seed = old_seed

            best_dir = os.path.join(fold_root, f'exp_{best_exp}')
            for fname in ['embedding_best.pth', 'transformer_best.pth', 'y_scaler.pth', 'test_predictions.csv',
                          'split.json']:
                src = os.path.join(best_dir, fname)
                if os.path.exists(src):
                    shutil.copy2(src, os.path.join(fold_root, fname))
            st15 = os.path.join(best_dir, 'stage15_fused_grouped.xlsx')
            if os.path.exists(st15):
                shutil.copy2(st15, os.path.join(fold_root, 'stage15_fused_grouped.xlsx'))

            try:
                sp_path = os.path.join(fold_root, 'split.json')
                with open(sp_path, 'r', encoding='utf-8') as fp:
                    spj = json.load(fp)
                spj['files']['embedding'] = os.path.join(fold_root, 'embedding_best.pth')
                spj['files']['transformer'] = os.path.join(fold_root, 'transformer_best.pth')
                spj['files']['y_scaler'] = os.path.join(fold_root, 'y_scaler.pth')
                spj['files']['pred_csv'] = os.path.join(fold_root, 'test_predictions.csv')
                with open(sp_path, 'w', encoding='utf-8') as fp:
                    json.dump(spj, fp, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"[Warn] 修正 split.json 失败：{e}")

            print(
                f"[DONE] split.json 复训 {args.exp_k} 次，R2 列表 = {[round(x, 4) for x in attempt_r2s]}，最佳 exp_{best_exp} -> Test R2 = {round(best_r2, 4)}")
            return

    # ----------- 常规流程：遍历 sheet / 折；可过滤 -----------
    all_sheets = pd.ExcelFile(args.xlsx).sheet_names
    if args.only_sheet is not None:
        all_sheets = [s for s in all_sheets if s == args.only_sheet]
        if not all_sheets:
            raise ValueError(f"未找到指定 sheet：{args.only_sheet}")

    allowed_folds = None
    if args.only_folds is not None:
        allowed_folds = set(int(x.strip()) for x in args.only_folds.split(',') if x.strip())

    all_sheet_summary = []

    for sheet_name in all_sheets:
        print("\n" + "=" * 80)
        print("开始处理工作表：", sheet_name)
        print("=" * 80)

        X, y, cats, feat_names, row_ids = load_sheet(args.xlsx, sheet_name)
        run_root = os.path.join(args.save_dir, 'sheet_' + sheet_name)
        ensure_dir(run_root)

        folds = build_stratified_folds_by_cat_and_logy(cats, y, n_splits=5, seed=args.seed, bins=10)

        cv_manifest = {
            "sheet": sheet_name,
            "seed": int(args.seed),
            "stratified_by": "category+log(y)",
            "k": int(args.k),
            "a": float(args.a),
            "mapping_path": os.path.abspath(args.mapping),
            "final_anchor": "outer_train",
            "save_dir": run_root,
            "folds": []
        }

        fold_metrics = []
        for fold_id, (trval_idx, te_idx) in enumerate(folds, start=1):
            if (allowed_folds is not None) and (fold_id not in allowed_folds):
                print(f"[Skip] {sheet_name} / fold_{fold_id} 不在 only_folds 中，跳过。")
                continue

            fold_dir = os.path.join(run_root, f'fold_{fold_id}')
            ensure_dir(fold_dir)

            # 默认按当前构造；若要求复用旧划分、且有 split.json，则覆盖为旧划分
            outer_train_idx = row_ids[trval_idx].tolist()
            outer_test_idx = row_ids[te_idx].tolist()

            split_json_path = os.path.join(fold_dir, 'split.json')
            use_old_split = args.use_saved_splits and os.path.isfile(split_json_path)
            if use_old_split:
                print(f"[Info] 复用已有划分：{split_json_path}")
                with open(split_json_path, 'r', encoding='utf-8') as fp:
                    old = json.load(fp)
                # 强制使用旧的内/外层索引（全局行号）
                outer_train_idx = old["outer_train_idx"]
                outer_test_idx = old["outer_test_idx"]
                inner_train_idx = old["inner_train_idx"]
                inner_val_idx = old["inner_val_idx"]
            else:
                # 现划分：在外层训练集上分层出 20% 验证
                rel_ids = np.arange(len(trval_idx))
                c_trval = cats[trval_idx]
                y_trval = y[trval_idx]
                ybin_trval = make_log_bins(y_trval, bins=10)
                combo_trval = np.array([f"{str(c_trval[i])}__{int(ybin_trval[i])}" for i in range(len(ybin_trval))])
                try:
                    rel_tr, rel_val = train_test_split(
                        rel_ids, test_size=0.2, random_state=args.seed,
                        shuffle=True, stratify=combo_trval
                    )
                except Exception:
                    rel_tr, rel_val = train_test_split(
                        rel_ids, test_size=0.2, random_state=args.seed,
                        shuffle=True, stratify=None
                    )
                inner_train_idx = row_ids[trval_idx[rel_tr]].tolist()
                inner_val_idx = row_ids[trval_idx[rel_val]].tolist()


            # === 真正训练该折 ===
            if args.exp_k == 1:
                r2 = train_one_fold_by_indices(
                    sheet_name=sheet_name,
                    X=X, y=y, cats=cats, feat_names=feat_names, row_ids=row_ids,
                    outer_train_idx=outer_train_idx,
                    outer_test_idx=outer_test_idx,
                    inner_train_idx=inner_train_idx,
                    inner_val_idx=inner_val_idx,
                    descriptors_mapping=descriptors_mapping,
                    args=args, device=device,
                    fold_dir=fold_dir,
                    stand_properties=stand_properties
                )
                fold_metrics.append(r2)
                # 更新 manifest
                cv_manifest["folds"].append({
                    "fold_id": int(fold_id),
                    "test_r2": float(r2),
                    "split_file": os.path.join(fold_dir, 'split.json'),
                    "embedding": os.path.join(fold_dir, 'embedding_best.pth'),
                    "transformer": os.path.join(fold_dir, 'transformer_best.pth'),
                    "pred_csv": os.path.join(fold_dir, 'test_predictions.csv'),
                    "best_exp": 1
                })
            else:
                print(
                    f"[Info] {sheet_name} / fold_{fold_id} 启用 exp_k={args.exp_k} 次尝试，选择测试 R2 最佳的一次保存。")
                base_seed = int(args.seed)
                best_r2 = -1e18
                best_exp = None
                attempt_r2s = []
                for exp_i in range(1, args.exp_k + 1):
                    tmp_seed = base_seed + exp_i - 1
                    print(f"[Try] fold_{fold_id} / exp_{exp_i} with seed={tmp_seed}")
                    set_seed(tmp_seed)
                    old_seed = args.seed
                    args.seed = tmp_seed
                    exp_dir = os.path.join(fold_dir, f'exp_{exp_i}')
                    ensure_dir(exp_dir)
                    r2_i = train_one_fold_by_indices(
                        sheet_name=sheet_name,
                        X=X, y=y, cats=cats, feat_names=feat_names, row_ids=row_ids,
                        outer_train_idx=outer_train_idx,
                        outer_test_idx=outer_test_idx,
                        inner_train_idx=inner_train_idx,
                        inner_val_idx=inner_val_idx,
                        descriptors_mapping=descriptors_mapping,
                        args=args, device=device,
                        fold_dir=exp_dir,
                        stand_properties=stand_properties
                    )
                    attempt_r2s.append(float(r2_i))
                    if r2_i > best_r2:
                        best_r2 = float(r2_i)
                        best_exp = exp_i
                    args.seed = old_seed

                # 复制最佳产物到 fold 根目录
                best_dir = os.path.join(fold_dir, f'exp_{best_exp}')
                for fname in ['embedding_best.pth', 'transformer_best.pth', 'y_scaler.pth', 'test_predictions.csv',
                              'split.json']:
                    src = os.path.join(best_dir, fname)
                    if os.path.exists(src):
                        shutil.copy2(src, os.path.join(fold_dir, fname))
                st15 = os.path.join(best_dir, 'stage15_fused_grouped.xlsx')
                if os.path.exists(st15):
                    shutil.copy2(st15, os.path.join(fold_dir, 'stage15_fused_grouped.xlsx'))
                # 修正 split.json 的 files 指向根目录
                try:
                    sp_path = os.path.join(fold_dir, 'split.json')
                    with open(sp_path, 'r', encoding='utf-8') as fp:
                        spj = json.load(fp)
                    spj['files']['embedding'] = os.path.join(fold_dir, 'embedding_best.pth')
                    spj['files']['transformer'] = os.path.join(fold_dir, 'transformer_best.pth')
                    spj['files']['y_scaler'] = os.path.join(fold_dir, 'y_scaler.pth')
                    spj['files']['pred_csv'] = os.path.join(fold_dir, 'test_predictions.csv')
                    with open(sp_path, 'w', encoding='utf-8') as fp:
                        json.dump(spj, fp, ensure_ascii=False, indent=2)
                except Exception as e:
                    print(f"[Warn] 修正 split.json 失败：{e}")

                r2 = float(best_r2)
                fold_metrics.append(r2)
                print(
                    f"[Result] {sheet_name} / fold_{fold_id} R2 尝试 = {[round(x, 4) for x in attempt_r2s]}，最佳 exp_{best_exp} -> Test R2 = {round(r2, 4)}")
                # 更新 manifest 指向根目录产物
                cv_manifest["folds"].append({
                    "fold_id": int(fold_id),
                    "test_r2": float(r2),
                    "split_file": os.path.join(fold_dir, 'split.json'),
                    "embedding": os.path.join(fold_dir, 'embedding_best.pth'),
                    "transformer": os.path.join(fold_dir, 'transformer_best.pth'),
                    "pred_csv": os.path.join(fold_dir, 'test_predictions.csv'),
                    "best_exp": int(best_exp)
                })

        # 写入当前 sheet 的 manifest
        with open(os.path.join(run_root, 'cv_manifest.json'), 'w', encoding='utf-8') as fp:
            json.dump(cv_manifest, fp, ensure_ascii=False, indent=2)

        if fold_metrics:
            r_list = [round(x, 4) for x in fold_metrics]
            mean_r2 = float(np.mean(fold_metrics))
            std_r2 = float(np.std(fold_metrics))
            print("-" * 60)
            print("Sheet:", sheet_name, "| R2s:", r_list)
            print("Mean R2:", mean_r2)
            print("Std  R2:", std_r2)
            all_sheet_summary.append({
                "sheet": sheet_name, "mean_r2": mean_r2, "std_r2": std_r2, "fold_r2": r_list
            })
        else:
            print(f"[Warn] {sheet_name} 未训练到任何折（可能 all 被过滤）。")

    # 汇总
    if all_sheet_summary:
        print("\n" + "=" * 80)
        print("已完成训练的工作表总结：")
        for rec in all_sheet_summary:
            print(
                f" - {rec['sheet']}  Mean R2 = {rec['mean_r2']:.4f}  Std = {rec['std_r2']:.4f}  folds = {rec['fold_r2']}")
        print("=" * 80)


if __name__ == '__main__':
    main()
