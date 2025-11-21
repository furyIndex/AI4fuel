
import os, shutil, sys, yaml, numpy as np, pandas as pd
from datetime import datetime

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import mean_squared_error, r2_score

apex_support = False
try:
    sys.path.append('./apex')
    from apex import amp
    apex_support = True
except:
    print("Please install apex for mixed precision training from: https://github.com/NVIDIA/apex")
    apex_support = False

def _save_config_file(model_checkpoints_folder):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        shutil.copy('./config_finetune.yaml', os.path.join(model_checkpoints_folder, 'config_finetune.yaml'))

class FineTune(object):
    def __init__(self, dataset, config):
        self.config = config
        self.device = self._get_device()
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        addon = ''
        if 'sheet' in config['dataset']: addon += f"_{config['dataset']['sheet']}"
        if hasattr(dataset, 'fold_id'): addon += f"_{dataset.fold_id}"
        dir_name = current_time + '_' + config['task_name'] + addon
        self.writer = SummaryWriter(log_dir=os.path.join('finetune', dir_name))
        self.dataset = dataset
        self.criterion = nn.MSELoss()

    def _get_device(self):
        if torch.cuda.is_available() and self.config['gpu'] != 'cpu':
            device = self.config['gpu']; torch.cuda.set_device(device)
        else:
            device = 'cpu'
        print("Running on:", device); return device

    def _load_pre_trained_weights(self, model):
        ckpt_cfg = self.config.get('fine_tune_from', '__scratch__')
        if not ckpt_cfg or ckpt_cfg == '__scratch__':
            print("[FT] fine_tune_from=__scratch__：不加载预训练，纯监督训练。")
            return model, False

        if ckpt_cfg.endswith('.pth') and os.path.isfile(ckpt_cfg):
            ckpt_path = ckpt_cfg
        elif os.path.isdir(ckpt_cfg):
            cand1 = os.path.join(ckpt_cfg, 'model.pth')
            cand2 = os.path.join(ckpt_cfg, 'checkpoints', 'model.pth')
            ckpt_path = cand1 if os.path.isfile(cand1) else cand2
        else:
            ckpt_path = os.path.join('./ckpt', ckpt_cfg, 'checkpoints', 'model.pth')

        if not os.path.exists(ckpt_path):
            print(f"[FT] 未找到预训练权重：{ckpt_path}，改为 Scratch。")
            return model, False

        state_dict = torch.load(ckpt_path, map_location='cpu')
        if hasattr(model, 'load_my_state_dict'):
            model.load_my_state_dict(state_dict)
            print(f"[FT] 通过 model.load_my_state_dict() 加载：{ckpt_path}")
        else:
            own = model.state_dict()
            matched = {k:v for k,v in state_dict.items() if (k in own and own[k].shape == v.shape)}
            own.update(matched); model.load_state_dict(own)
            print(f"[FT] 部分加载（matched={len(matched)}）：{ckpt_path}")
        return model, True


    def _make_optimizer(self, model):
        head_names = ('graph_pred', 'pred', 'head')
        head_params, base_params = [], []
        for n, p in model.named_parameters():
            if any(k in n for k in head_names):
                head_params.append(p)
            else:
                base_params.append(p)

        if self.config.get('freeze_backbone', False):
            for p in base_params: p.requires_grad = False
            print("[FT] 冻结骨干，仅训练头部。")

        if len(head_params) == 0:
            print("[FT] 未识别到 head 参数，使用单一学习率。")
            return torch.optim.Adam(
                model.parameters(), self.config['init_lr'],
                weight_decay=eval(self.config['weight_decay'])
            )

        return torch.optim.Adam(
            [
                {'params': base_params, 'lr': self.config.get('init_base_lr', self.config['init_lr'])},
                {'params': head_params, 'lr': self.config['init_lr']},
            ],
            weight_decay=eval(self.config['weight_decay'])
        )

    def _step(self, model, data):
        __, pred = model(data)                 # (B,1)
        loss = self.criterion(pred.view(-1, 1), data.y.view(-1, 1))
        return loss

    def train_one_fold(self):
        train_loader, valid_loader, test_loader = self.dataset.get_data_loaders()

        # 模型
        if self.config['model_type'] == 'gin':
            from models.ginet_finetune import GINet
            model = GINet('regression', **self.config["model"]).to(self.device)
        elif self.config['model_type'] == 'gcn':
            from models.gcn_finetune import GCN
            model = GCN('regression', **self.config["model"]).to(self.device)
        else:
            raise ValueError('Undefined GNN model.')

        model, _ = self._load_pre_trained_weights(model)
        optimizer = self._make_optimizer(model)

        if apex_support and self.config['fp16_precision']:
            model, optimizer = amp.initialize(model, optimizer, opt_level='O2', keep_batchnorm_fp32=True)

        ckpt_dir = os.path.join(self.writer.log_dir, 'checkpoints')
        _save_config_file(ckpt_dir)

        n_iter = 0; best_val_rmse = np.inf
        for epoch in range(self.config['epochs']):
            model.train()
            for _, data in enumerate(train_loader):
                optimizer.zero_grad()
                data = data.to(self.device)
                loss = self._step(model, data)
                if apex_support and self.config['fp16_precision']:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                optimizer.step()

                if n_iter % self.config['log_every_n_steps'] == 0:
                    self.writer.add_scalar('train_loss', loss.item(), global_step=n_iter)
                n_iter += 1

            if epoch % self.config['eval_every_n_epochs'] == 0:
                val_rmse = self._evaluate_rmse(model, valid_loader)
                self.writer.add_scalar('val_RMSE', val_rmse, global_step=epoch)
                if val_rmse < best_val_rmse:
                    best_val_rmse = val_rmse
                    torch.save(model.state_dict(), os.path.join(ckpt_dir, 'model.pth'))

        self._test_and_save(model, test_loader)

    @torch.no_grad()
    def _evaluate_rmse(self, model, loader):
        model.eval()
        preds, labels = [], []
        for data in loader:
            data = data.to(self.device)
            __, pred = model(data)
            preds.append(pred.cpu().numpy()); labels.append(data.y.cpu().numpy())
        preds = np.concatenate(preds).reshape(-1); labels = np.concatenate(labels).reshape(-1)
        return mean_squared_error(labels, preds, squared=False)

    @torch.no_grad()
    def _test_and_save(self, model, test_loader):
        state_dict = torch.load(os.path.join(self.writer.log_dir, 'checkpoints', 'model.pth'),
                                map_location=self.device)
        model.load_state_dict(state_dict); model.eval()

        preds, labels, rows = [], [], []
        for data in test_loader:
            data = data.to(self.device)
            __, pred = model(data)
            preds.append(pred.cpu().numpy()); labels.append(data.y.cpu().numpy())
            if hasattr(data, "row_id"):
                rows += data.row_id.cpu().tolist()

        preds = np.concatenate(preds).reshape(-1); labels = np.concatenate(labels).reshape(-1)
        rmse = mean_squared_error(labels, preds, squared=False)
        r2   = r2_score(labels, preds)
        print(f"[TEST] RMSE {rmse:.6f}  R2 {r2:.4f}")

        out_csv = os.path.join(self.writer.log_dir, "test_predictions.csv")
        pd.DataFrame({"row_id": rows if rows else list(range(len(preds))),
                      "y_true": labels, "y_pred": preds}).to_csv(out_csv, index=False)
        print("Saved:", out_csv)

def _collect_sheet_list(cfg):
    ds = cfg['dataset']
    if 'sheets' in ds and ds['sheets']:
        return list(ds['sheets'])
    if str(ds.get('sheet', '')).upper() == 'ALL':
        xlsx = ds['xlsx']
        try:
            xls = pd.ExcelFile(xlsx)
            return [s for s in xls.sheet_names]
        except Exception as e:
            raise RuntimeError(f"读取 {xlsx} 的 sheet 名失败: {e}")
    if 'sheet' in ds and ds['sheet']:
        return [ds['sheet']]
    raise ValueError("请在 config_finetune.yaml 里提供 dataset.sheets 列表，或设 dataset.sheet: ALL/具体名字。")

def main(cfg):
    from dataset.dataset_excel_splits_finetune import MolExcelFinetuneDatasetWrapper as Wrapper

    sheets = _collect_sheet_list(cfg)
    print("[INFO] 将依次训练这些 sheet：", sheets)

    all_rows = []
    for sheet in sheets:

        from glob import glob
        splits_root = cfg['dataset']['splits_root']
        fold_paths = []
        for pat in [os.path.join(splits_root, sheet, "fold_*", "split.json"),
                    os.path.join(splits_root, f"sheet_{sheet}", "fold_*", "split.json")]:
            fold_paths = sorted(glob(pat)) or fold_paths
        if not fold_paths:
            print(f"[WARN] 跳过：未找到 {sheet} 的 split.json（在 {splits_root} 下）。")
            continue

        sheet_rows = []
        for spath in fold_paths:
            fold_id = os.path.basename(os.path.dirname(spath))
            print(f"\n==== Sheet: {sheet} | Fold: {fold_id} ====")
            sub_cfg = dict(cfg); sub_cfg['dataset'] = dict(cfg['dataset'])
            sub_cfg['dataset']['sheet'] = sheet
            sub_cfg['dataset']['fold_id'] = fold_id

            dataset = Wrapper(cfg['batch_size'], **sub_cfg['dataset'])
            ft = FineTune(dataset, sub_cfg)
            ft.train_one_fold()
            pred_csv = os.path.join(ft.writer.log_dir, "test_predictions.csv")
            if os.path.exists(pred_csv):
                df = pd.read_csv(pred_csv)
                r2 = r2_score(df["y_true"].values, df["y_pred"].values)
                row = {"sheet": sheet, "fold": fold_id, "r2": float(r2)}
                sheet_rows.append(row); all_rows.append(row)


        if sheet_rows:
            out_dir = os.path.join("finetune", f"sheet_{sheet}_ft")
            os.makedirs(out_dir, exist_ok=True)
            pd.DataFrame(sheet_rows).to_csv(os.path.join(out_dir, "summary.csv"), index=False)
            print("Saved:", os.path.join(out_dir, "summary.csv"))


    if all_rows:
        os.makedirs("finetune", exist_ok=True)
        pd.DataFrame(all_rows).to_csv(os.path.join("finetune", "summary_all_sheets_ft.csv"), index=False)
        print("Saved:", os.path.join("finetune", "summary_all_sheets_ft.csv"))

if __name__ == "__main__":
    config = yaml.load(open("./config_finetune.yaml", "r"), Loader=yaml.FullLoader)
    config['dataset']['task'] = 'regression'
    print(config)
    main(config)
