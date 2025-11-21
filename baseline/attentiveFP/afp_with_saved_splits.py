
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import json, argparse, random, time
import numpy as np
import pandas as pd
from collections import Counter

import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from rdkit import Chem

from other_baseline.attentiveFP.afp_model.AttentiveLayers import Fingerprint
from other_baseline.attentiveFP.afp_model.getFeatures import get_smiles_array, save_smiles_dicts




from pyGPGO.covfunc import matern32
from pyGPGO.acquisition import Acquisition
from pyGPGO.surrogates.GaussianProcess import GaussianProcess
from pyGPGO.GPGO import GPGO



def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    try: torch.use_deterministic_algorithms(True)
    except Exception: pass

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def per_class_counts(arr):
    c = Counter(list(arr)); return {k:int(v) for k,v in c.items()}

def read_sheet(xlsx_path, sheet):
    df = pd.read_excel(xlsx_path, sheet_name=sheet)
    if 'SMILES' not in df.columns:
        raise ValueError(f"[{sheet}] 未找到 'SMILES' 列")
    cat_col = df.columns[0]; label_col = df.columns[-2]

    def to_cano(s):
        m = Chem.MolFromSmiles(str(s).strip())
        return Chem.MolToSmiles(m, canonical=True, isomericSmiles=True) if m is not None else np.nan

    smiles_cano = df['SMILES'].map(to_cano)
    out = pd.DataFrame({
        'category': df[cat_col].astype(str),
        'smiles': smiles_cano,
        'label': pd.to_numeric(df[label_col], errors='coerce')
    })
    out = out.dropna(subset=['smiles']).reset_index(drop=False).rename(columns={'index':'row_id'})
    out['row_id'] = out['row_id'].astype(int)
    return out

def deep_update(dst, src):
    for k, v in src.items():
        if isinstance(v, dict) and k in dst and isinstance(dst[k], dict):
            dst[k].update(v)
        else:
            dst[k] = v
    return dst

class AFPDataset(torch.utils.data.Dataset):
    def __init__(self, smiles_list, y):
        self.smiles = list(smiles_list)
        self.y = np.asarray(y, dtype=np.float32).reshape(-1, 1)
    def __len__(self): return len(self.smiles)
    def __getitem__(self, idx): return self.smiles[idx], self.y[idx]




def make_collate(feature_dicts, cache_prefix):
    def _collate(batch):
        smiles_batch = [b[0] for b in batch]
        y_batch = torch.from_numpy(np.vstack([b[1] for b in batch])).float()

        def in_cache(s): return s in feature_dicts.get('smiles_to_atom_mask', {})

        missing = [s for s in smiles_batch if not in_cache(s)]
        if missing:
            extra_prefix = cache_prefix + "_extra"
            new_dicts = save_smiles_dicts(list(set(missing)), extra_prefix)
            deep_update(feature_dicts, new_dicts)
            missing2 = [s for s in smiles_batch if not in_cache(s)]
            if missing2:
                raise KeyError(f"仍缺特征：{missing2[:10]}{' 等' if len(missing2)>10 else ''}")

        pack = get_smiles_array(smiles_batch, feature_dicts)
        if isinstance(pack, (list, tuple)) and len(pack) >= 5:
            x_atom, x_bonds, x_atom_index, x_bond_index, x_mask = pack[:5]
        else:
            raise ValueError("Unexpected return from get_smiles_array")

        xa = torch.from_numpy(np.asarray(x_atom)).float()
        xb = torch.from_numpy(np.asarray(x_bonds)).float()
        xa_idx = torch.from_numpy(np.asarray(x_atom_index)).long()
        xb_idx = torch.from_numpy(np.asarray(x_bond_index)).long()
        xmask = torch.from_numpy(np.asarray(x_mask)).float()
        return [xa, xb, xa_idx, xb_idx, xmask, y_batch]
    return _collate

def train_val_once(tr_loader, va_loader, device, hp, max_epochs, patience):

    with torch.no_grad():
        xb = next(iter(tr_loader))
    atom_fdim = xb[0].shape[-1]; bond_fdim = xb[1].shape[-1]

    model = Fingerprint(
        radius=int(round(hp['radius'])),
        T=int(round(hp['T'])),
        input_feature_dim=atom_fdim,
        input_bond_dim=bond_fdim,
        fingerprint_dim=int(round(hp['fingerprint_dim'])),
        output_units_num=1,
        p_dropout=float(hp['p_dropout'])
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=10**(-float(hp['learning_rate'])),
        weight_decay=10**(-float(hp['weight_decay']))
    )
    criterion = nn.MSELoss()

    best_state, best_val_r2, best_val_mse = None, -1e9, 1e18
    best_epoch, no_improve = -1, 0

    for epoch in range(1, 1 + max_epochs):
        model.train()
        for xa, xb, xa_idx, xb_idx, xmask, y in tr_loader:
            xa, xb, xa_idx, xb_idx, xmask, y = [t.to(device) for t in (xa, xb, xa_idx, xb_idx, xmask, y)]
            optimizer.zero_grad()
            pred = model(xa, xb, xa_idx, xb_idx, xmask)[1]
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            y_val, y_hat = [], []
            for xa, xb, xa_idx, xb_idx, xmask, y in va_loader:
                xa, xb, xa_idx, xb_idx, xmask = [t.to(device) for t in (xa, xb, xa_idx, xb_idx, xmask)]
                pred = model(xa, xb, xa_idx, xb_idx, xmask)[1].cpu().numpy().ravel()
                y_hat.append(pred); y_val.append(y.numpy().ravel())
            y_val = np.concatenate(y_val); y_hat = np.concatenate(y_hat)
            val_r2 = r2_score(y_val, y_hat)
            val_mse = mean_squared_error(y_val, y_hat)

        improved = (val_r2 > best_val_r2)
        if improved:
            best_val_r2 = float(val_r2)
            best_val_mse = float(val_mse)
            best_epoch = int(epoch)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    return best_val_mse, best_val_r2, best_epoch, best_state

def train_one_fold_with_hpo(tr_df, va_df, te_df, device, base_args, out_dir, feature_dicts, cache_prefix,
                            max_iter=30, init_evals=2, random_seed=168):
    ensure_dir(out_dir)


    scaler = StandardScaler().fit(tr_df[['label']].values.astype(np.float32))
    y_tr = scaler.transform(tr_df[['label']].values.astype(np.float32))
    y_va = scaler.transform(va_df[['label']].values.astype(np.float32))
    y_te = scaler.transform(te_df[['label']].values.astype(np.float32))


    tr_ds = AFPDataset(tr_df['smiles'].values, y_tr)
    va_ds = AFPDataset(va_df['smiles'].values,   y_va)
    te_ds = AFPDataset(te_df['smiles'].values,   y_te)

    collate = make_collate(feature_dicts, cache_prefix)
    tr_loader = DataLoader(tr_ds, batch_size=base_args['batch_size'], shuffle=True,  collate_fn=collate)
    va_loader = DataLoader(va_ds, batch_size=base_args['batch_size'], shuffle=False, collate_fn=collate)
    te_loader = DataLoader(te_ds, batch_size=base_args['batch_size'], shuffle=False, collate_fn=collate)


    hpo_csv = os.path.join(out_dir, "hpo_trials.csv")
    if not os.path.exists(hpo_csv):
        with open(hpo_csv, 'w', encoding='utf-8') as f:
            f.write('radius,T,fingerprint_dim,p_dropout,weight_decay_exp,learning_rate_exp,best_epoch,val_mse,val_r2,time\n')


    cov = matern32(); gp = GaussianProcess(cov); acq = Acquisition(mode='UCB')
    param = {
        'radius': ('int', [2, 6]),
        'T': ('int', [1, 5]),
        'fingerprint_dim': ('int', [30, 300]),
        'weight_decay': ('cont', [2, 6]),
        'learning_rate': ('cont', [2, 5]),
        'p_dropout': ('cont', [0, 0.5]),
    }


    def f(radius, T, fingerprint_dim, weight_decay, learning_rate, p_dropout, direction=False):
        hp = dict(radius=radius, T=T, fingerprint_dim=fingerprint_dim,
                  weight_decay=weight_decay, learning_rate=learning_rate, p_dropout=p_dropout)
        val_mse, val_r2, best_epoch, _ = train_val_once(
            tr_loader, va_loader, device, hp,
            max_epochs=base_args['max_epochs'], patience=base_args['patience']
        )
        with open(hpo_csv, 'a', encoding='utf-8') as ff:
            ff.write(','.join([
                str(int(round(radius))), str(int(round(T))), str(int(round(fingerprint_dim))),
                f"{p_dropout:.6f}", f"{weight_decay:.6f}", f"{learning_rate:.6f}",
                str(int(best_epoch)), f"{val_mse:.8f}", f"{val_r2:.8f}", time.strftime('%Y-%m-%d %H:%M:%S')
            ]) + '\n')
        return -val_mse if not direction else val_mse

    np.random.seed(random_seed)
    gpgo = GPGO(gp, acq, f, param)
    gpgo.run(max_iter=max_iter, init_evals=init_evals)
    hp_opt, perf_opt = gpgo.getResult()


    best_hparams = {
        "atom_layers": int(round(hp_opt['radius'])),
        "mol_layers": int(round(hp_opt['T'])),
        "fp_dim": int(round(hp_opt['fingerprint_dim'])),
        "dropout_p": float(hp_opt['p_dropout']),
        "lr": 10**(-float(hp_opt['learning_rate'])),
        "weight_decay": 10**(-float(hp_opt['weight_decay'])),
        "batch_size": base_args['batch_size'],
        "max_epochs": base_args['max_epochs'],
        "patience": base_args['patience'],
    }


    with open(os.path.join(out_dir, "hpo_best.json"), 'w', encoding='utf-8') as f:
        json.dump({
            "hp_opt_raw": hp_opt,
            "best_hparams_numeric": best_hparams,
            "perf_opt_neg_val_mse": float(perf_opt)
        }, f, ensure_ascii=False, indent=2)


    _, _, _, best_state = train_val_once(tr_loader, va_loader, device, hp_opt,
                                         max_epochs=base_args['max_epochs'], patience=base_args['patience'])


    with torch.no_grad():
        xb = next(iter(tr_loader))
    atom_fdim = xb[0].shape[-1]; bond_fdim = xb[1].shape[-1]
    final_model = Fingerprint(
        radius=best_hparams['atom_layers'], T=best_hparams['mol_layers'],
        input_feature_dim=atom_fdim, input_bond_dim=bond_fdim,
        fingerprint_dim=best_hparams['fp_dim'], output_units_num=1,
        p_dropout=best_hparams['dropout_p']
    ).to(device)
    final_model.load_state_dict(best_state)


    final_model.eval()
    with torch.no_grad():
        y_te_std, yhat_te_std = [], []
        for xa, xb, xa_idx, xb_idx, xmask, y in te_loader:
            xa, xb, xa_idx, xb_idx, xmask = [t.to(device) for t in (xa, xb, xa_idx, xb_idx, xmask)]
            pred = final_model(xa, xb, xa_idx, xb_idx, xmask)[1].cpu().numpy().ravel()
            yhat_te_std.append(pred); y_te_std.append(y.numpy().ravel())
        y_te_std = np.concatenate(y_te_std); yhat_te_std = np.concatenate(yhat_te_std)
        y_te_real = scaler.inverse_transform(y_te_std.reshape(-1,1)).ravel()
        yhat_te_real = scaler.inverse_transform(yhat_te_std.reshape(-1,1)).ravel()
        te_r2 = float(r2_score(y_te_real, yhat_te_real))


    torch.save(final_model, os.path.join(out_dir, 'attentivefp_best.pth'))
    with open(os.path.join(out_dir, 'best_params.json'), 'w', encoding='utf-8') as f:
        json.dump({
            "atom_layers": best_hparams['atom_layers'],
            "mol_layers": best_hparams['mol_layers'],
            "fp_dim": best_hparams['fp_dim'],
            "dropout": best_hparams['dropout_p'],
            "lr": best_hparams['lr'],
            "weight_decay": best_hparams['weight_decay'],
            "batch_size": best_hparams['batch_size'],
            "max_epochs": best_hparams['max_epochs'],
            "patience": best_hparams['patience']
        }, f, ensure_ascii=False, indent=2)

    pd.DataFrame({
        "row_id": te_df['row_id'].astype(int).values,
        "category": te_df['category'].values,
        "y_true": y_te_real.astype(float),
        "y_pred": yhat_te_real.astype(float)
    }).to_csv(os.path.join(out_dir, 'test_predictions.csv'), index=False)

    return te_r2, best_hparams


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--xlsx', required=True)
    ap.add_argument('--splits_root', required=True)
    ap.add_argument('--save_dir', default='./cv_runs_afp_hpo')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--seed', type=int, default=42)

    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--max_epochs', type=int, default=800)
    ap.add_argument('--patience', type=int, default=50)

    ap.add_argument('--hpo_max_iter', type=int, default=30)
    ap.add_argument('--hpo_init_evals', type=int, default=2)
    ap.add_argument('--hpo_seed', type=int, default=168)

    ap.add_argument('--debug_overfit', type=int, default=0)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    ensure_dir(args.save_dir)

    sheet_names = pd.ExcelFile(args.xlsx).sheet_names
    for sheet in sheet_names:
        print("\n" + "="*80); print(f"开始处理 sheet: {sheet}"); print("="*80)

        df = read_sheet(args.xlsx, sheet)
        run_src = os.path.join(args.splits_root, f"sheet_{sheet}")
        run_dst = os.path.join(args.save_dir, f"sheet_{sheet}")
        ensure_dir(run_dst)

        folds = sorted([d for d in os.listdir(run_src) if d.startswith("fold_") and os.path.isdir(os.path.join(run_src,d))],
                       key=lambda s: int(s.split('_')[-1]))

        all_smiles_sheet = sorted(df['smiles'].unique().tolist())
        sheet_feature_cache = os.path.join(run_dst, "afp_features_sheet")
        feature_dicts = save_smiles_dicts(all_smiles_sheet, sheet_feature_cache)
        print(f"feature dicts file saved as {sheet_feature_cache}.pickle")

        fold_metrics = []
        manifest = {"sheet": sheet, "seed": int(args.seed), "stratified_by": "category", "save_dir": run_dst, "folds": []}

        for fold_name in folds:
            print("fold_name:", fold_name)
            src_fold = os.path.join(run_src, fold_name)
            dst_fold = os.path.join(run_dst, fold_name)
            ensure_dir(dst_fold)

            split_path = os.path.join(src_fold, "split.json")
            with open(split_path, 'r', encoding='utf-8') as f:
                sp = json.load(f)

            outer_train_idx = sp["outer_train_idx"]
            inner_train_idx = sp["inner_train_idx"]
            inner_val_idx   = sp["inner_val_idx"]
            outer_test_idx  = sp["outer_test_idx"]

            if args.debug_overfit > 0 and len(inner_train_idx) > args.debug_overfit:
                inner_train_idx = inner_train_idx[:args.debug_overfit]
                print(f"[DEBUG] Overfit mode: use first {args.debug_overfit} samples for inner_train.")

            tr_df = df.iloc[inner_train_idx].reset_index(drop=True)
            va_df = df.iloc[inner_val_idx].reset_index(drop=True)
            te_df = df.iloc[outer_test_idx].reset_index(drop=True)

            per_all = per_class_counts(df['category'].values)
            per_outer = per_class_counts(te_df['category'].values)

            base_args = dict(batch_size=args.batch_size, max_epochs=args.max_epochs, patience=args.patience)

            te_r2, best_hp = train_one_fold_with_hpo(
                tr_df, va_df, te_df, device, base_args, dst_fold, feature_dicts, sheet_feature_cache,
                max_iter=args.hpo_max_iter, init_evals=args.hpo_init_evals, random_seed=args.hpo_seed
            )
            fold_metrics.append(te_r2)
            print(f"{fold_name}: Test R2 = {round(te_r2,4)}")


            split_out = {
                "sheet": sheet, "seed": int(args.seed), "stratified_by": "category",
                "per_class_all": per_all, "per_class_outer_test": per_outer,
                "outer_train_idx": outer_train_idx, "outer_test_idx": outer_test_idx,
                "inner_train_idx": inner_train_idx, "inner_val_idx": inner_val_idx,
                "best_params": {
                    "atom_layers": best_hp['atom_layers'], "mol_layers": best_hp['mol_layers'],
                    "fp_dim": best_hp['fp_dim'], "dropout": best_hp['dropout_p'],
                    "lr": best_hp['lr'], "weight_decay": best_hp['weight_decay'],
                    "batch_size": args.batch_size, "max_epochs": args.max_epochs, "patience": args.patience
                },
                "files": {
                    "attentivefp": os.path.join(dst_fold, "attentivefp_best.pth"),
                    "pred_csv": os.path.join(dst_fold, "test_predictions.csv"),
                    "best_params": os.path.join(dst_fold, "best_params.json"),
                    "hpo_trials": os.path.join(dst_fold, "hpo_trials.csv"),
                    "hpo_best": os.path.join(dst_fold, "hpo_best.json")
                }
            }
            with open(os.path.join(dst_fold, "split.json"), 'w', encoding='utf-8') as f:
                json.dump(split_out, f, ensure_ascii=False, indent=2)

            manifest["folds"].append({
                "fold_id": int(fold_name.split('_')[-1]),
                "test_r2": float(te_r2),
                "split_file": os.path.join(dst_fold, "split.json"),
                "attentivefp": os.path.join(dst_fold, "attentivefp_best.pth"),
                "pred_csv": os.path.join(dst_fold, "test_predictions.csv")
            })

        with open(os.path.join(run_dst, "cv_manifest.json"), 'w', encoding='utf-8') as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

        r_list = [round(x,4) for x in fold_metrics]
        print("-"*60)
        print(f"Sheet {sheet}: fold R2 = {r_list}"
              f" | Mean = {round(float(np.mean(fold_metrics)), 4) if fold_metrics else 'nan'}"
              f" | Std = {round(float(np.std(fold_metrics)), 4) if fold_metrics else 'nan'}")

if __name__ == "__main__":
    main()
