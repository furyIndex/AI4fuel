
import os
import numpy as np
import datetime
from typing import Sequence, Dict, Optional, List, Any, Tuple, Callable, Union
from argparse import ArgumentParser
import pandas as pd
import math
import sklearn.metrics
import matplotlib.pyplot as plt
import torch
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
from torch.utils.tensorboard.writer import SummaryWriter
from torch.optim.lr_scheduler import MultiStepLR
from torch.nn.utils.clip_grad import clip_grad_norm_

from metrics_and_losses import (raw_mse, mse_skipnan, mse_skipnan_with_stddev, bce_skipnan,
                                    per_prop_mse, per_prop_stable_mse, per_prop_r2, per_prop_roc_auc)
from other_baseline.acs.dataset import DatasetProcessor, Split, Purpose
from other_baseline.acs.performance_analysis import performance_analysis, k_fold_analysis
from other_baseline.acs.model import ModelWrapper
from checkpointing import Checkpointing
import shutil
import glob
import deepchem as dc
from deepchem.splits.splitters import ScaffoldSplitter
from sklearn.utils import shuffle



def _find_smiles_col(df: pd.DataFrame) -> str:
    for c in df.columns:
        if str(c).strip().lower() == 'smiles':
            return c
    raise ValueError("未在表中找到 'SMILES' 列。")


def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def materialize_multitask_fold_for_acs(
        xlsx_path: str,
        splits_root: str,
        sheet_names: Optional[List[str]],
        fold_id: int,
        out_dir: str
) -> Tuple[str, np.ndarray, np.ndarray, np.ndarray, List[str]]:

    os.makedirs(out_dir, exist_ok=True)
    xl = pd.ExcelFile(xlsx_path)
    if sheet_names is None:
        sheet_names = xl.sheet_names


    per_sheet = {}  # name -> dict(df, smiles_col, label_col, splits)
    union_train, union_val, union_test = set(), set(), set()

    for s in sheet_names:
        df = xl.parse(s)
        df = _normalize_cols(df)
        smi_col = _find_smiles_col(df)
        y_col = df.columns[-2]

        df = df.drop_duplicates(subset=[smi_col], keep='first').reset_index(drop=True)


        sp = os.path.join(splits_root, f"sheet_{s}", f"fold_{fold_id}", "split.json")
        if not os.path.isfile(sp):
            raise FileNotFoundError(f"未找到 split.json: {sp}")
        import json
        with open(sp, 'r', encoding='utf-8') as f:
            jj = json.load(f)
        tri = list(map(int, jj['inner_train_idx']))
        vai = list(map(int, jj['inner_val_idx']))
        tei = list(map(int, jj['outer_test_idx']))

        train_smi = set(df.iloc[tri][smi_col].astype(str))
        val_smi = set(df.iloc[vai][smi_col].astype(str))
        test_smi = set(df.iloc[tei][smi_col].astype(str))

        union_train |= train_smi
        union_val |= val_smi
        union_test |= test_smi

        per_sheet[s] = dict(df=df, smi_col=smi_col, y_col=y_col)


    union_val -= union_test
    union_train -= (union_test | union_val)

    all_smiles = sorted(list(union_train | union_val | union_test))
    mt_df = pd.DataFrame({'SMILES': all_smiles})

    for s, pack in per_sheet.items():
        df, smi_col, y_col = pack['df'], pack['smi_col'], pack['y_col']
        sub = df[[smi_col, y_col]].copy()
        sub.columns = ['SMILES', s]

        sub[s] = pd.to_numeric(sub[s], errors='coerce')
        mt_df = mt_df.merge(sub, on='SMILES', how='left')


    tmp_xlsx = os.path.join(out_dir, f"multitask_fold_{fold_id}.xlsx")
    with pd.ExcelWriter(tmp_xlsx, engine='openpyxl') as writer:
        mt_df.to_excel(writer, sheet_name='Sheet1', index=False)


    smiles2idx = {smi: i for i, smi in enumerate(mt_df['SMILES'].astype(str))}
    train_idx = np.array([smiles2idx[smi] for smi in union_train if smi in smiles2idx], dtype=int)
    val_idx = np.array([smiles2idx[smi] for smi in union_val if smi in smiles2idx], dtype=int)
    test_idx = np.array([smiles2idx[smi] for smi in union_test if smi in smiles2idx], dtype=int)


    train_idx.sort();
    val_idx.sort();
    test_idx.sort()

    task_names = list(per_sheet.keys())
    return tmp_xlsx, train_idx, val_idx, test_idx, task_names




def deterministic_train_val_split(all_data_size: int,
                                  num_folds: int,
                                  fold_id: int,
                                  random_seed: int = 123
                                  ) -> Tuple[np.ndarray, np.ndarray]:
    fold_id = int(fold_id % num_folds)
    fold_size = all_data_size // num_folds
    rng = np.random.default_rng(random_seed)
    all_indices = rng.permutation(all_data_size)
    val_indices = all_indices[fold_id * fold_size:(fold_id + 1) * fold_size]
    train_indices = all_indices[~np.isin(all_indices, val_indices)]
    return train_indices, val_indices


def deterministic_deepchem_murcko_scaffold_stratified_train_val_split(
        all_data_size: int,
        num_folds: int,
        fold_id: int,
        rng_seed: int = 123,
        series_smiles: pd.Series = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    scaffold_splitter = ScaffoldSplitter()
    dummy_features = np.zeros(len(series_smiles))
    series_smiles = shuffle(series_smiles.tolist(), random_state=rng_seed)
    dataset = dc.data.DiskDataset.from_numpy(
        X=dummy_features, y=np.zeros(len(series_smiles)), w=np.zeros(len(series_smiles)), ids=series_smiles)
    train_size, valid_size, test_size = 0.8, 0.1, 0.1
    train_dataset, valid_dataset, test_dataset = scaffold_splitter.train_valid_test_split(
        dataset, frac_train=train_size, frac_valid=valid_size, frac_test=test_size, seed=rng_seed)
    train_inds = np.array([series_smiles.index(smile) for smile in train_dataset.ids])
    valid_inds = np.array([series_smiles.index(smile) for smile in valid_dataset.ids])
    test_inds = np.array([series_smiles.index(smile) for smile in test_dataset.ids])
    return train_inds, valid_inds, test_inds


class Trainer:

    def __init__(self,
                 data_path: Optional[str] = None,
                 validate_first: Optional[bool] = None,
                 tag: Optional[str] = None,
                 debug: bool = False,
                 model_type: str = 'acs',
                 filters: Optional[str] = None,
                 num_folds: int = 1,
                 limit_folds: int = 1,
                 fold_id: int = 0,
                 datetime_str: datetime = None,
                 max_steps: int = 50_000,
                 val_interval: int = 500,
                 ablated_props: list = [],
                 delete_checkpoints: int = False,
                 mlp_multiplier: int = 1,
                 random_seed: int = 123,
                 learning_rate: float = 1e-3,
                 weight_decay: float = 1e-4,
                 conv_layers: int = 8,
                 use_scaffold_split: bool = False,
                 dropout_rate: float = 0.0,
                 all_aggregation: bool = True,
                 batch_size: int = 20,
                 smoothing_alpha: float = 0.5,
                 task_type: str = 'regression',
                 adaptive_checkpointing: bool = False,
                 global_loss_checkpointing: bool = False,
                 # ===== 新增：用于外部提供划分 =====
                 provided_train_idx: Optional[np.ndarray] = None,
                 provided_val_idx: Optional[np.ndarray] = None,
                 provided_test_idx: Optional[np.ndarray] = None) -> None:

        self.model_type = model_type
        self.datetime_str = datetime_str
        self.fold_id = fold_id
        self.num_folds = num_folds
        self.limit_folds = limit_folds
        self.val_interval = val_interval
        self.delete_checkpoints = delete_checkpoints
        self.tag: Optional[str] = tag
        self.debug: bool = debug
        self.debug_iters = 2
        self.max_steps = max_steps
        self.mlp_multiplier = mlp_multiplier
        self.random_seed = random_seed
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.conv_layers = conv_layers
        self.use_scaffold_split = use_scaffold_split
        self.dropout_rate = dropout_rate
        self.all_aggregation = all_aggregation
        self.batch_size = batch_size
        self.adaptive_checkpointing = adaptive_checkpointing
        self.global_loss_checkpointing = global_loss_checkpointing
        self.smoothing_alpha = smoothing_alpha
        self.task_type = task_type

        self.validate_first: bool
        if validate_first is None:
            self.validate_first = False
        else:
            self.validate_first = validate_first

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"device={self.device}")

        data_path = data_path if data_path is not None else f"data/SAF.xlsx"
        self.all_data = DatasetProcessor(data_path, "Sheet1",
                                         self.task_type, filters=filters,
                                         cache_data=False, ablated_props=ablated_props)

        self.property_names = list(self.all_data.df_properties.columns)

        if (provided_train_idx is not None) and (provided_val_idx is not None):
            train_indices = provided_train_idx
            val_indices = provided_val_idx
            test_indices = provided_test_idx
        elif self.use_scaffold_split:
            train_indices, val_indices, test_indices = deterministic_deepchem_murcko_scaffold_stratified_train_val_split(
                len(self.all_data), self.num_folds, self.fold_id, random_seed, self.all_data.series_smiles)
        else:
            train_indices, val_indices = deterministic_train_val_split(
                len(self.all_data), self.num_folds, self.fold_id, random_seed)
            test_indices = None

        oversample_factor = 1000 if self.task_type == 'regression' else 1
        self.train_data = Split(self.all_data, train_indices, self.task_type, purpose=Purpose.Train,
                                oversample_factor=oversample_factor).as_dataset()
        self.train_for_eval_data = Split(self.all_data, train_indices, self.task_type, purpose=Purpose.Val,
                                         oversample_factor=oversample_factor).as_dataset()
        self.val_data = Split(self.all_data, val_indices, self.task_type, purpose=Purpose.Val,
                              oversample_factor=oversample_factor).as_dataset()

        if test_indices is not None and len(test_indices) > 0:
            self.test_data = Split(self.all_data, test_indices, self.task_type, purpose=Purpose.Test,
                                   oversample_factor=oversample_factor).as_dataset()
        else:
            self.test_data = None

        batch_size = self.batch_size
        val_batch_size = max(1, batch_size // 2)
        num_workers = 8

        self.train_dl = DataLoader(
            self.train_data, batch_size=batch_size,
            shuffle=True, num_workers=num_workers,
            persistent_workers=num_workers > 0)

        self.train_for_eval_dl = DataLoader(
            self.train_for_eval_data, batch_size=val_batch_size,
            shuffle=False, num_workers=num_workers)

        self.val_dl = DataLoader(
            self.val_data, batch_size=val_batch_size,
            shuffle=False, num_workers=num_workers)

        if self.test_data is not None:
            self.test_dl = DataLoader(
                self.test_data, batch_size=val_batch_size,
                shuffle=False, num_workers=num_workers)

        sample_data = self.train_data[0]
        num_props = sample_data.properties.shape[-1]
        node_feature_size = sample_data.x.shape[-1]
        edge_feature_size = sample_data.edge_attr.shape[-1]

        print("model_type is", model_type)
        print("max_steps is", max_steps)
        print("ablated_props are", ablated_props)
        print("delete_checkpoints is", delete_checkpoints)
        print("mlp_multiplier is", mlp_multiplier)
        print("random_seed is", random_seed)
        print("learning_rate is", learning_rate)
        print("weight_decay is", weight_decay)
        print("conv_layers is", conv_layers)
        print("use_scaffold_split is", use_scaffold_split)
        print("dropout_rate is", dropout_rate)
        print("all_aggregation is", all_aggregation)
        print("batch_size is", batch_size)
        print("val_interval is", val_interval)
        print("task_type is", task_type)
        print("adaptive_checkpointing is", adaptive_checkpointing)
        print("global_loss_checkpointing is", global_loss_checkpointing)

        self.model: ModelWrapper = ModelWrapper(task_type, model_type, node_feature_size, edge_feature_size,
                                                num_props, mlp_multiplier, conv_layers, dropout_rate, all_aggregation)
        self.model.to(self.device)
        print(self.model)

        num_trainable_params = sum([p.numel() for p in self.model.parameters()])
        print('num_trainable_params = ' + str(num_trainable_params))

        self.logger: Optional[SummaryWriter] = None
        self.iteration: Optional[int] = None

        self.loss_fn = mse_skipnan_with_stddev if self.task_type == 'regression' else bce_skipnan

    def train(self) -> None:
        print(f"Start training run {self.fold_id + 1} out",
              f"of {self.limit_folds} based on {self.num_folds} folds")

        self.lr = self.learning_rate
        self.weight_decay = self.weight_decay
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        max_steps = self.max_steps
        val_interval = self.val_interval
        milestones = [int(max_steps * v) for v in (0.6, 0.7, 0.8, 0.9)]
        lr_scheduler = MultiStepLR(self.optimizer, milestones=milestones, gamma=1 / math.sqrt(10))

        self.art_dir_name = (f"{self.datetime_str}" + (f"_{self.tag}" if self.tag is not None else ""))
        self.artefact_dir = os.path.join("runs", self.art_dir_name, f"fold_{self.fold_id + 1}")
        os.makedirs(self.artefact_dir, exist_ok=True)

        if self.logger is None:
            self.logger = SummaryWriter(self.artefact_dir, flush_secs=30)

        if self.model_type in ['acs', 'mtl-glc', 'stl']:
            self.checkpointing = Checkpointing(self.property_names, self.model, self.artefact_dir, self.model_type,
                                               self.smoothing_alpha)
        else:
            self.checkpointing = None

        self.iteration = 0
        if self.validate_first:
            self.validate()

        self.stop_training = False
        while True:
            if self.stop_training:
                break

            print(f"{self.iteration=}")
            for _, batch in enumerate(self.train_dl):
                self.model.train()
                batch = batch.to(self.device)
                pred = self.model(batch)
                target = batch.properties
                target_stddev = batch.property_stddevs
                assert pred.shape == target.shape, f"{pred.shape=} {target.shape=}"
                loss = self.loss_fn(pred, target, target_stddev) if self.task_type == 'regression' else self.loss_fn(
                    pred, target)

                self.optimizer.zero_grad()
                loss.backward()
                grad_clip_val = 100.0
                grad_norm = clip_grad_norm_(self.model.parameters(), grad_clip_val).item()
                self.optimizer.step()

                train_print_interval = self.debug_iters if self.debug else 20
                if self.iteration % train_print_interval == train_print_interval - 1:
                    learning_rate = lr_scheduler.get_last_lr()[0]
                    self.logger.add_scalar("train/learning_rate", learning_rate, self.iteration)
                    self.logger.add_scalar(f"train/grad_norm", grad_norm, self.iteration)
                    print(f"step={self.iteration}, loss={loss.item():.6f}")
                    self.logger.add_scalar("train/loss", loss.item(), self.iteration)

                    raw_loss = raw_mse(pred, target)
                    per_prop_loss = torch.nanmean(raw_loss, dim=0)
                    columns = list(self.all_data.df_properties.columns)
                    values = per_prop_loss.detach().cpu().numpy()
                    for column, value in zip(columns, values):
                        self.logger.add_scalar(f"train/loss/{column}", value, self.iteration)

                val_interval = self.debug_iters if self.debug else val_interval
                if self.iteration % val_interval == val_interval - 1:
                    self.validate()
                    pred_np = pred.detach().cpu().numpy()
                    target_np = target.detach().cpu().numpy()
                    self.dump_scatters(pred_np, target_np, "train")

                lr_scheduler.step()

                self.iteration += 1
                if self.iteration >= max_steps or self.stop_training:
                    self.stop_training = True
                    break
        print("Done")

    def validate(self, mode='val', test=False):

        if (self.checkpointing is not None) and test:
            print("Training finished, reloading best checkpoint for each sub-network")
            for i in range(len(self.property_names)):
                self.checkpointing.reload_checkpoint(i)

        self.model.eval()

        pred_list, target_list, raw_loss_list, loss_list = [], [], [], []

        if mode == "val":
            loader = self.val_dl
        elif mode == "train_for_eval":
            loader = self.train_for_eval_dl
        elif mode == "test":
            if hasattr(self, 'test_dl'):
                loader = self.test_dl
            else:
                return

        for i_batch, batch in enumerate(loader):
            if self.debug and i_batch >= self.debug_iters:
                break

            batch = batch.to(self.device)
            with torch.no_grad():
                if (self.model_type in ['acs', 'mtl-glc']) and test:
                    pred = self.specialized_backbone_load_then_eval(batch)
                else:
                    pred = self.model(batch)
                target = batch.properties
                target_stddev = batch.property_stddevs
                loss = self.loss_fn(pred, target, target_stddev) if self.task_type == 'regression' else self.loss_fn(
                    pred, target)

            pred_list.append(pred.cpu())
            target_list.append(target)
            raw_loss_list.append(raw_mse(pred.cpu(), target.cpu()))
            loss_list.append(loss.item())

        total_loss = np.mean(np.array(loss_list)).item()
        print(f"VALIDATION step={self.iteration}, loss={total_loss:.6f}")

        block_pred = torch.cat(pred_list, dim=0).cpu().numpy()
        block_target = torch.cat(target_list, dim=0).cpu().numpy()

        mse_list = per_prop_mse(block_pred, block_target)
        block_raw_loss = torch.cat(raw_loss_list, dim=0)
        per_prop_loss = torch.nanmean(block_raw_loss, dim=0)
        columns = list(self.all_data.df_properties.columns)
        values = per_prop_loss.numpy()
        for column, value, mse in zip(columns, values, mse_list):
            print(f"{column.rjust(10)} loss={value:.3f} mse={mse:.3f}")

        stable_mse_list = per_prop_stable_mse(block_pred, block_target)
        total_stable_mse = np.nanmean(np.array(stable_mse_list)).item()

        if mode == "val":
            split_indices = self.val_data.get_indices()
        elif mode == "train_for_eval":
            split_indices = self.train_for_eval_data.get_indices()
        elif mode == "test" and hasattr(self, 'test_data') and (self.test_data is not None):
            split_indices = self.test_data.get_indices()
        else:
            split_indices = None

        if split_indices is not None:
            val_rows = self.all_data.df_features.iloc[split_indices]
        else:
            val_rows = None

        self.dump_scatters(block_pred, block_target, mode)

        if (self.checkpointing is not None) and not test:
            self.stop_training = self.checkpointing(block_pred, block_target, total_loss)
            num_unfrozen_params = sum([p.numel() for p in self.model.parameters() if p.requires_grad])
            if num_unfrozen_params > 0:
                self.optimizer = torch.optim.AdamW((p for p in self.model.parameters() if p.requires_grad),
                                                   lr=self.lr, weight_decay=self.weight_decay)

        if self.logger is not None:
            self.logger.add_scalar("val/loss", total_loss, self.iteration)
            for column, value in zip(columns, values):
                self.logger.add_scalar(f"val/loss/{column}", value, self.iteration)
            for column, st_mse in zip(columns, stable_mse_list):
                self.logger.add_scalar(f"val/stable_mse_90/{column}", st_mse, self.iteration)
            self.logger.add_scalar("val/stable_mse_90", total_stable_mse, self.iteration)

        if self.artefact_dir is not None:
            ckpt_name = f"latest_{self.fold_id + 1}.pth"
            torch.save(self.model.state_dict(), os.path.join(self.artefact_dir, ckpt_name))

        if test and self.task_type == 'regression':
            prop_list = list(self.all_data.df_properties.columns)
            performance_analysis(self.task_type, self.artefact_dir, prop_list, val_rows,
                                 block_pred, block_target, mode, normalized=True, fold_id=self.fold_id)
            performance_analysis(self.task_type, self.artefact_dir, prop_list, val_rows,
                                 self.all_data.normalizer.decode(block_pred),
                                 self.all_data.normalizer.decode(block_target),
                                 mode, normalized=False, fold_id=self.fold_id)
        elif test and self.task_type == 'classification':
            prop_list = list(self.all_data.df_properties.columns)
            performance_analysis(self.task_type, self.artefact_dir, prop_list, val_rows,
                                 block_pred, block_target, mode, normalized=True, fold_id=self.fold_id)

    def specialized_backbone_load_then_eval(self, batch):
        gms_pred_list = []
        for i in range(len(self.property_names)):
            self.checkpointing.reload_backbone(i)
            gms_pred_list.append(self.model(batch)[:, i].unsqueeze(1))
        return torch.cat(gms_pred_list, dim=1)

    def dump_scatters(self, block_pred: np.ndarray, block_target: np.ndarray, suffix: str):
        print(f"------------- Per-property metrics for {suffix} -------------")
        property_names = list(self.all_data.df_properties.columns)
        property_full_names = self.all_data.property_full_names

        mse_list = per_prop_mse(block_pred, block_target)
        stable_mse_list = per_prop_stable_mse(block_pred, block_target)
        r2_list = per_prop_r2(block_pred, block_target, is_stable=False)
        stable_r2_list = per_prop_r2(block_pred, block_target, is_stable=True)
        if self.task_type == 'classification':
            roc_auc_list = per_prop_roc_auc(block_pred, block_target)

        if self.task_type == 'regression':
            for i_prop, (reg_mse, st_mse, r2, stable_r2) in enumerate(
                    zip(mse_list, stable_mse_list, r2_list, stable_r2_list)):
                print(f"{property_names[i_prop].rjust(10)} "
                      f"mse={reg_mse:.3f}, rmse={math.sqrt(reg_mse):.3f}, "
                      f"stable_mse={st_mse:.3f}, r2={r2:.3f} stable_r2={stable_r2:.3f}")
        else:
            for i_prop, auc_roc in enumerate(roc_auc_list):
                print(f"{property_names[i_prop].rjust(10)} auc_roc={auc_roc:.3f}")

        ncols = 8
        fig, axs = plt.subplots(64 // ncols, ncols, figsize=(24, 14))
        for i in range(block_pred.shape[1]):
            pred = block_pred[:, i]
            target = block_target[:, i]
            name = property_names[i]
            ax = axs[i // ncols, i % ncols]
            ax.scatter(target, pred, s=4)
            ax.set_title(f"{name} MSE={mse_list[i]:.3f} StableMSE={stable_mse_list[i]:.3f}\n"
                         f"R2={r2_list[i]:.3f} StableR2={stable_r2_list[i]:.3f}\n"
                         f"({property_full_names[name]})", fontsize=8)
            ax.set_xlabel('target');
            ax.set_ylabel('pred');
            ax.grid()
        plt.tight_layout()

        img_name = f"scatters_{suffix}_{self.fold_id + 1}.png"
        img_path = os.path.join(self.artefact_dir, img_name) if self.artefact_dir is not None else img_name
        plt.savefig(img_path)


        if suffix == 'test' and self.artefact_dir is not None:
            property_names = list(self.all_data.df_properties.columns)

            n_test = np.sum(~np.isnan(block_target), axis=0).astype(int)  # 需要把 block_target 作为参数传入或在 validate() 里保存
            df = pd.DataFrame({
                'property': property_names,
                'n_test': n_test,
                'mse': mse_list,
                'stable_mse': stable_mse_list,
                'r2': r2_list,
                'stable_r2': stable_r2_list
            })
            df.to_csv(os.path.join(self.artefact_dir, f'metrics_test_{self.fold_id + 1}.csv'), index=False)



    def load_snapshot(self, snapshot_path: str):
        self.model.load_state_dict(torch.load(snapshot_path, map_location=self.device))

    def test(self):
        self.validate(mode='val', test=True)
        self.validate(mode='train_for_eval', test=True)
        self.validate(mode='test', test=True)
        if self.delete_checkpoints and (self.checkpointing is not None):
            shutil.rmtree((os.path.join(self.artefact_dir, "checkpoints")))
            pth_file = next(glob.iglob(os.path.join(self.artefact_dir, '*.pth')), None);
            pth_file and os.remove(pth_file)
            print("Deleted checkpoints")


def main():
    parser = ArgumentParser(description='ACS trainer (adapted for custom splits)')
    parser.add_argument('--data-path', '-d', type=str, help='all_data.xlsx')
    parser.add_argument('--test-snapshot', type=str, help='Provide .pth, get test results')
    parser.add_argument('--start-from-pth', type=str, help='In training mode, start with the provided .pth')
    parser.add_argument('--validate-first', action='store_true', help="Run validation before training starts")
    parser.add_argument('--tag', type=str, help='Suffix for artefact dir name')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--filters', type=str, default=None)
    parser.add_argument('--model-type', '-m', type=str, default='acs',
                        help='acs / mtl / mtl-glc / stl')
    parser.add_argument('--task-type', '-tt', type=str, default='regression',
                        help='regression or classification')
    parser.add_argument('--num-folds', type=int, default=5)
    parser.add_argument('--limit-folds', type=int, default=None)
    parser.add_argument('--max-steps', '-steps', dest='max_steps', type=int, default=50_000)
    parser.add_argument('--parallel-folds', '-pf', dest='parallel_folds', type=int, default=0)
    parser.add_argument('--val-interval', '-vi', dest='val_interval', type=int, default=500)
    parser.add_argument('--ablated-props', '-ap', nargs='*', type=str, default='None')
    parser.add_argument('--delete-checkpoints', '-dc', action='store_true')
    parser.add_argument('--mlp-multiplier', '-mm', type=float, default=1)
    parser.add_argument('--random-seed', '-rs', type=int, default=42)
    parser.add_argument('--learning-rate', '-lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', '-wd', type=float, default=1e-4)
    parser.add_argument('--conv-layers', '-cl', type=int, default=8)
    parser.add_argument('--use-scaffold-split', '-ss', action='store_true', default=False)
    parser.add_argument('--dropout-rate', '-dr', type=float, default=0.3)
    parser.add_argument('--all-aggregation', '-aa', type=str, default="0")
    parser.add_argument('--batch-size', '-bs', type=int, default=20)
    parser.add_argument('--smoothing-alpha', '-sa', type=float, default=0.1)




    parser.add_argument('--splits-root', type=str, default=None,
                        help='父目录：sheet_<name>/fold_<k>/split.json')
    parser.add_argument('--sheets', nargs='*', default=None,
                        help='指定要纳入多任务的 sheet 名（默认：xlsx 内所有 sheet）')
    parser.add_argument('--fold-id', type=int, default=None,
                        help='使用指定 fold（1-based）；若未提供则用原来的 K 折循环')

    args = parser.parse_args()

    adaptive_checkpointing, global_loss_checkpointing = False, False
    if args.model_type in ['acs', 'stl']:
        adaptive_checkpointing = True
    elif args.model_type == 'mtl-glc':
        global_loss_checkpointing = True

    if args.limit_folds is None: args.limit_folds = args.num_folds
    datetime_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


    if args.data_path is None:
        args.data_path = f"data/SAF.xlsx"
    else:
        if args.data_path in ('sider', 'tox21', 'clintox'):
            args.data_path = f"data/{args.data_path}.csv"
        elif not args.data_path.lower().endswith(('.xlsx', '.csv')):
            args.data_path = 'data/' + args.data_path + '.xlsx'


    if args.all_aggregation == "1":
        args.all_aggregation = True
    elif args.all_aggregation == "0":
        args.all_aggregation = False


    def _run_one_fold(fold_zero_based: int):

        fold_1b = fold_zero_based + 1

        tmp_dir = os.path.join("runs", f"{datetime_str}_materialized")
        tmp_xlsx, tr_idx, va_idx, te_idx, tasks = materialize_multitask_fold_for_acs(
            xlsx_path=args.data_path,
            splits_root=args.splits_root,
            sheet_names=args.sheets,
            fold_id=fold_1b,
            out_dir=tmp_dir
        )
        print(f"[INFO] Multitask dataset for fold {fold_1b} written to: {tmp_xlsx}")
        trainer = Trainer(
            data_path=tmp_xlsx,
            validate_first=args.validate_first,
            tag=args.tag,
            debug=args.debug,
            model_type=args.model_type,
            filters=args.filters,
            num_folds=1,
            limit_folds=1,
            fold_id=fold_zero_based,
            datetime_str=datetime_str,
            max_steps=args.max_steps,
            val_interval=args.val_interval,
            ablated_props=args.ablated_props,
            delete_checkpoints=args.delete_checkpoints,
            mlp_multiplier=args.mlp_multiplier,
            random_seed=args.random_seed,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            conv_layers=args.conv_layers,
            use_scaffold_split=False,
            dropout_rate=args.dropout_rate,
            all_aggregation=args.all_aggregation,
            batch_size=args.batch_size,
            smoothing_alpha=args.smoothing_alpha,
            task_type=args.task_type,
            adaptive_checkpointing=adaptive_checkpointing,
            global_loss_checkpointing=global_loss_checkpointing,
            provided_train_idx=tr_idx,
            provided_val_idx=va_idx,
            provided_test_idx=te_idx
        )
        if args.test_snapshot is not None:
            trainer.load_snapshot(args.test_snapshot)
            trainer.test()
        else:
            if args.start_from_pth is not None:
                trainer.load_snapshot(args.start_from_pth)
                print(f"Loading snapshot from {args.start_from_pth}")
            trainer.train()
            trainer.test()

    if args.splits_root is not None:
        if args.fold_id is not None:
            _run_one_fold(args.fold_id - 1)
        else:

            for k in range(args.limit_folds):
                _run_one_fold(k)
        print("Done")
        return


    if args.parallel_folds > 0:
        trainer = Trainer(
            data_path=args.data_path,
            validate_first=args.validate_first,
            tag=args.tag,
            debug=args.debug,
            model_type=args.model_type,
            filters=args.filters,
            num_folds=args.num_folds,
            limit_folds=1,
            fold_id=args.parallel_folds - 1,
            datetime_str=datetime_str,
            max_steps=args.max_steps,
            val_interval=args.val_interval,
            ablated_props=args.ablated_props,
            delete_checkpoints=args.delete_checkpoints,
            mlp_multiplier=args.mlp_multiplier,
            random_seed=args.random_seed,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            conv_layers=args.conv_layers,
            use_scaffold_split=args.use_scaffold_split,
            dropout_rate=args.dropout_rate,
            all_aggregation=args.all_aggregation,
            batch_size=args.batch_size,
            smoothing_alpha=args.smoothing_alpha,
            task_type=args.task_type,
            adaptive_checkpointing=adaptive_checkpointing,
            global_loss_checkpointing=global_loss_checkpointing
        )
        if args.test_snapshot is not None:
            trainer.load_snapshot(args.test_snapshot);
            trainer.test()
        else:
            if args.start_from_pth is not None:
                trainer.load_snapshot(args.start_from_pth);
                print(f"Loading snapshot from {args.start_from_pth}")
            trainer.train();
            trainer.test()
    else:
        for fold_id in range(args.limit_folds):
            trainer = Trainer(
                data_path=args.data_path,
                validate_first=args.validate_first,
                tag=args.tag,
                debug=args.debug,
                model_type=args.model_type,
                filters=args.filters,
                num_folds=args.num_folds,
                limit_folds=args.limit_folds,
                fold_id=fold_id,
                datetime_str=datetime_str,
                max_steps=args.max_steps,
                val_interval=args.val_interval,
                ablated_props=args.ablated_props,
                delete_checkpoints=args.delete_checkpoints,
                mlp_multiplier=args.mlp_multiplier,
                random_seed=args.random_seed,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                conv_layers=args.conv_layers,
                use_scaffold_split=args.use_scaffold_split,
                dropout_rate=args.dropout_rate,
                all_aggregation=args.all_aggregation,
                batch_size=args.batch_size,
                smoothing_alpha=args.smoothing_alpha,
                task_type=args.task_type,
                adaptive_checkpointing=adaptive_checkpointing,
                global_loss_checkpointing=global_loss_checkpointing
            )
            if args.test_snapshot is not None:
                trainer.load_snapshot(args.test_snapshot);
                trainer.test()
            else:
                if args.start_from_pth is not None:
                    trainer.load_snapshot(args.start_from_pth);
                    print(f"Loading snapshot from {args.start_from_pth}")
                trainer.train();
                trainer.test()
        k_fold_analysis(datetime_str, limit_folds=args.limit_folds)
    print("Done")


if __name__ == "__main__":
    main()
