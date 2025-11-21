from argparse import Namespace
import csv
from logging import Logger
import os
from typing import List

import numpy as np
from tensorboardX import SummaryWriter
import torch
import pickle
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader

from .evaluate import evaluate, evaluate_predictions
from .predict import predict
from .train import train
from other_baseline.KANO.chemprop_local.data import StandardScaler
from other_baseline.KANO.chemprop_local.data.utils import get_class_sizes, get_data, get_task_names, split_data
from other_baseline.KANO.chemprop_local.models import build_model, build_pretrain_model, add_functional_prompt
from other_baseline.KANO.chemprop_local.nn_utils import param_count
from other_baseline.KANO.chemprop_local.utils import build_optimizer, build_lr_scheduler, get_loss_func, get_metric_func, load_checkpoint,\
    makedirs, save_checkpoint
from other_baseline.KANO.chemprop_local.parsing import parse_train_args, modify_train_args
from other_baseline.KANO.chemprop_local.torchlight import initialize_exp

# === XLSX 适配：从 xlsx + split.json 物化出 train/val/test CSV ===
try:
    from other_baseline.KANO.kano_excel_splits import materialize_fold_to_csv
except Exception:
    materialize_fold_to_csv = None


def run_training(args: Namespace, prompt: bool, logger: Logger = None) -> List[float]:
    """
    Trains a model and returns test scores on the model checkpoint with the highest validation score.
    """
    if logger is not None:
        debug, info = logger.debug, logger.info
    else:
        debug = info = print

    # Set GPU
    if args.gpu is not None and args.gpu >= 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        args.cuda = True
    else:
        args.cuda = False

    # ---- XLSX→CSV 适配（固定划分） ----
    if str(args.data_path).lower().endswith('.xlsx'):
        if materialize_fold_to_csv is None:
            raise RuntimeError("未找到 tools/kano_excel_splits.py，请把它放到仓库根目录 tools/ 下。")
        sheet = getattr(args, 'xlsx_sheet', None) or os.getenv('KANO_SHEET', None)
        split_json = getattr(args, 'split_json', None) or os.getenv('KANO_SPLIT_JSON', None)
        fold_id = int(getattr(args, 'fold_id', os.getenv('KANO_FOLD', '1')))

        if split_json is None:
            split_json = os.path.join(os.path.dirname(args.data_path), 'split.json')
        if not os.path.isfile(split_json):
            raise FileNotFoundError(f'split.json 未找到：{split_json}')

        train_csv, val_csv, test_csv = materialize_fold_to_csv(
            xlsx_path=args.data_path, sheet_name=sheet, split_json_path=split_json,
            fold_id=fold_id, out_dir=args.save_dir
        )
        args.separate_val_path = val_csv
        args.separate_test_path = test_csv
        args.data_path = train_csv
        args.split_type = 'random'
        args.split_sizes = (1.0, 0.0, 0.0)
        info(f"[Excel Adapter] Sheet={sheet or '<default>'} | Fold={fold_id} | CSVs -> {os.path.dirname(train_csv)}")

    # ================= 原训练流程 =================
    info('Loading data')
    args.task_names = get_task_names(args.data_path)
    data = get_data(path=args.data_path, args=args, logger=logger)
    args.num_tasks = data.num_tasks()
    args.features_size = data.features_size()
    info(f'Number of tasks = {args.num_tasks}')

    # Split data
    if args.separate_test_path:
        test_data = get_data(path=args.separate_test_path, args=args,
                             features_path=args.separate_test_features_path, logger=logger)
    if args.separate_val_path:
        val_data = get_data(path=args.separate_val_path, args=args,
                            features_path=args.separate_val_features_path, logger=logger)

    if args.separate_val_path and args.separate_test_path:
        train_data = data
    elif args.separate_val_path:
        train_data, _, test_data = split_data(data=data, split_type=args.split_type,
                                              sizes=(0.8, 0.2, 0.0), seed=args.seed, args=args, logger=logger)
    elif args.separate_test_path:
        train_data, val_data, _ = split_data(data=data, split_type=args.split_type,
                                             sizes=(0.8, 0.2, 0.0), seed=args.seed, args=args, logger=logger)
    else:
        train_data, val_data, test_data = split_data(data=data, split_type=args.split_type,
                                                     sizes=args.split_sizes, seed=args.seed, args=args, logger=logger)

    # 保存测试集 SMILES/targets 以便写预测
    test_smiles, test_targets = test_data.smiles(), test_data.targets()

    # Feature scaling
    if args.features_scaling:
        features_scaler = train_data.normalize_features(replace_nan_token=0)
        val_data.normalize_features(features_scaler)
        test_data.normalize_features(features_scaler)
    else:
        features_scaler = None

    args.train_data_size = len(train_data)
    # 修正：每次直接拟合一个 scaler，不再用 load_checkpoint 读 scaler.pt
    if args.dataset_type == 'regression':
        scaler = StandardScaler()
        train_targets = train_data.targets()
        scaler.fit(train_targets)
    else:
        scaler = None

    loss_func = get_loss_func(args)
    metric_func = get_metric_func(metric=args.metric)

    # Ensemble 容器
    if args.dataset_type == 'multiclass':
        sum_test_preds = np.zeros((len(test_smiles), args.num_tasks, args.multiclass_num_classes))
    else:
        sum_test_preds = np.zeros((len(test_smiles), args.num_tasks))

    for model_idx in range(args.ensemble_size):
        save_dir = os.path.join(args.save_dir, f'model_{model_idx}')
        makedirs(save_dir)
        try:
            writer = SummaryWriter(log_dir=save_dir)
        except:
            writer = SummaryWriter(logdir=save_dir)

        if args.checkpoint_path is not None:
            model = build_model(args, encoder_name=args.encoder_name)
            model.encoder.load_state_dict(torch.load(args.checkpoint_path, map_location='cpu'), strict=False)
        else:
            model = build_model(args, encoder_name=args.encoder_name)

        if args.step == 'functional_prompt':
            add_functional_prompt(model, args)

        if args.cuda:
            model = model.cuda()

        save_checkpoint(os.path.join(save_dir, 'model.pt'), model, scaler, features_scaler, args)

        optimizer = build_optimizer(model, args)
        scheduler = build_lr_scheduler(optimizer, args)

        best_score = float('inf') if args.minimize_score else -float('inf')
        n_iter = 0

        for epoch in range(args.epochs):
            n_iter = train(
                model=model,
                prompt=prompt,
                data=train_data,
                loss_func=loss_func,
                optimizer=optimizer,
                scheduler=scheduler,
                args=args,
                n_iter=n_iter,
                logger=logger,
                writer=writer
            )
            if isinstance(scheduler, ExponentialLR):
                scheduler.step()

            # 验证
            val_scores = evaluate(model=model, prompt=prompt, data=val_data,
                                  num_tasks=args.num_tasks, metric_func=metric_func,
                                  dataset_type=args.dataset_type, logger=logger, batch_size=args.batch_size)
            avg_val_score = np.nanmean(val_scores)

            # 测试（仅记录）
            test_preds = predict(model=model, prompt=prompt, data=test_data,
                                 batch_size=args.batch_size, scaler=scaler)
            test_scores = evaluate_predictions(
                preds=test_preds, targets=test_targets, num_tasks=args.num_tasks,
                metric_func=metric_func, dataset_type=args.dataset_type, logger=logger
            )
            avg_test_score = np.nanmean(test_scores)

            is_best = (avg_val_score < best_score) if args.minimize_score else (avg_val_score > best_score)
            if is_best:
                best_score = avg_val_score
                save_checkpoint(os.path.join(save_dir, 'model.pt'), model, scaler, features_scaler, args)

        # 用最佳 ckpt 做最终测试 + 存预测
        model = load_checkpoint(os.path.join(save_dir, 'model.pt'))
        test_preds = predict(model=model, prompt=prompt, data=test_data,
                             batch_size=args.batch_size, scaler=scaler)
        sum_test_preds += np.array(test_preds)

        import pandas as pd
        df_pred = pd.DataFrame(test_preds)
        if df_pred.shape[1] == 1:
            df_pred.columns = ['prediction']
        df_pred.insert(0, 'smiles', test_smiles)
        df_pred.to_csv(os.path.join(save_dir, 'test_preds.csv'), index=False)

    # Ensemble 评估 + 输出到根目录
    avg_test_preds = (sum_test_preds / args.ensemble_size).tolist()
    ensemble_scores = evaluate_predictions(
        preds=avg_test_preds,
        targets=test_targets,
        num_tasks=args.num_tasks,
        metric_func=metric_func,
        dataset_type=args.dataset_type,
        logger=logger
    )

    import pandas as pd
    ens_df = pd.DataFrame(avg_test_preds)
    if ens_df.shape[1] == 1:
        ens_df.columns = ['prediction']
    ens_df.insert(0, 'smiles', test_smiles)
    ens_df.to_csv(os.path.join(args.save_dir, 'test_preds.csv'), index=False)

    return ensemble_scores