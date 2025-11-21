
import sys

from torch.nn.functional import mse_loss


import os

from train_kfold.kfold_scaler import LogTargetScaler, inv_if_scaler

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import argparse
import json
import os
import random
import shutil

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import optuna
from collections import Counter
from sklearn.model_selection import StratifiedKFold, KFold

from model.embedding_net import EmbeddingMLP, ContrastiveLoss, embedding_val_func
from descriptors_group.getGCNDescriptors import knn_train, l1_norm, norm_adj_train, knn_val, norm_adj_val
from model.transformerModel import SimpleTransformerRegressor





def to_tensor(x):
    return torch.tensor(x, dtype=torch.float32)


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def embed(model, X, device):
    x_s = model._x_scaler.transform(X)
    with torch.no_grad():
        e = model(to_tensor(x_s).to(device)).cpu().numpy()
    return e



def save_stage15_to_xlsx(writer, sheet_name, categories, labels, X_seq, group_order):

    n = X_seq.shape[0]
    seq_len = X_seq.shape[1]
    data = {"category": list(categories), "label": [float(v) for v in labels]}
    import json as _json
    for gi in range(seq_len):
        col_name = str(group_order[gi]) if group_order is not None else f"group_{gi}"
        col_vals = []
        for r in range(n):
            arr = [float(x) for x in X_seq[r, gi, :].tolist()]
            col_vals.append(_json.dumps(arr, ensure_ascii=False))
        data[col_name] = col_vals
    pd.DataFrame(data).to_excel(writer, sheet_name=sheet_name, index=False)





def train_embedding_stage(X_tr, y_tr, X_val, y_val, input_dim, device,
                          max_epochs=200,
                          patience=15,
                          batch_size=256, lr=1e-2, weight_decay=1e-3, margin=3.0):
    x_scaler = StandardScaler()
    X_tr_s = x_scaler.fit_transform(X_tr)
    X_val_s = x_scaler.transform(X_val)

    y_scaler = StandardScaler()
    y_tr_s = y_scaler.fit_transform(y_tr.reshape(-1, 1)).ravel()
    y_val_s = y_scaler.transform(y_val.reshape(-1, 1)).ravel()

    tr_loader = DataLoader(TensorDataset(to_tensor(X_tr_s), to_tensor(y_tr_s.reshape(-1, 1))),
                           batch_size=batch_size, shuffle=True)

    model = EmbeddingMLP(input_dim, 512, 128).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = ContrastiveLoss(margin=margin).to(device)

    best_state = None
    best_spearman = -1.0
    epochs_no_improve = 0

    for epoch in range(max_epochs):
        model.train()
        for xb, yb in tr_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            emb = model(xb)
            loss = criterion(emb, yb)
            loss.backward()
            optimizer.step()


        model.eval()
        with torch.no_grad():
            pred = model(to_tensor(X_val_s).to(device))
            sp = embedding_val_func(pred, to_tensor(y_val_s.reshape(-1, 1)).to(device))
            sp_val = float(sp[0]) if isinstance(sp, (list, tuple, np.ndarray)) else float(sp)


        if sp_val > best_spearman:
            best_spearman = sp_val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch + 1}. Best Spearman: {best_spearman:.4f}")
                break

        if epoch % 20 == 0 or epochs_no_improve == 0:
            print(f"Epoch {epoch + 1}: Spearman = {sp_val:.4f} (best: {best_spearman:.4f})")

    print(f"Stage 1 training finished. Best Spearman: {best_spearman:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    model._x_scaler = x_scaler
    model._y_scaler = y_scaler
    return model




def build_gcn_features(X_tr, X_val, X_te, feat_names, model, device, k=10, a=0.1,
                       grouped=True, mapping=None):

    E_tr = embed(model, X_tr, device)
    E_val = embed(model, X_val, device)
    E_te = embed(model, X_te, device)

    S_tr = np.dot(E_tr, E_tr.T)
    S_val = np.dot(E_val, E_tr.T)
    S_te = np.dot(E_te, E_tr.T)

    tr_knn = knn_train(S_tr, k)
    tr_norm = l1_norm(tr_knn)
    A_tr, A_tilde_tr, deg_tr = norm_adj_train(tr_norm, a)

    X_tr_s = model._x_scaler.transform(X_tr).astype(np.float32)
    X_val_s = model._x_scaler.transform(X_val).astype(np.float32)
    X_te_s = model._x_scaler.transform(X_te).astype(np.float32)

    n_tr, n_val, n_te = X_tr_s.shape[0], X_val_s.shape[0], X_te_s.shape[0]

    Xtr_fused = np.dot(A_tilde_tr.astype(np.float32), X_tr_s)



    tr_block_val = np.hstack([A_tr.astype(np.float32), np.zeros((n_tr, n_val), dtype=np.float32)])
    val_block = knn_val(S_val, n_val, n_tr + n_val, k).astype(np.float32)
    full_A_val = np.vstack([tr_block_val, val_block])
    A_val = norm_adj_val(full_A_val, deg_tr, a)
    X_stack_val = np.vstack([X_tr_s, X_val_s]).astype(np.float32)
    Xval_fused = np.dot(A_val, X_stack_val)[-n_val:]

    tr_block_te = np.hstack([A_tr.astype(np.float32), np.zeros((n_tr, n_te), dtype=np.float32)])
    te_block = knn_val(S_te, n_te, n_tr + n_te, k).astype(np.float32)
    full_A_te = np.vstack([tr_block_te, te_block])
    A_te = norm_adj_val(full_A_te, deg_tr, a)
    X_stack_te = np.vstack([X_tr_s, X_te_s]).astype(np.float32)
    Xte_fused = np.dot(A_te, X_stack_te)[-n_te:]

    if grouped:
        if mapping is None:
            raise ValueError("Grouped=True need descriptors mapping JSON。")
        max_cols = max(len(v) for v in mapping.values())
        group_order = list(mapping.keys())
        name2idx = {nm: i for i, nm in enumerate(feat_names)}

        def to_seq(mat):
            rows = []
            for r in range(mat.shape[0]):
                seq = []
                for g in group_order:
                    vals = [float(mat[r, name2idx[c]]) if c in name2idx else 0.0 for c in mapping.get(g, [])]
                    while len(vals) < max_cols:
                        vals.append(0.0)
                    seq.append(vals)
                rows.append(seq)
            return np.array(rows, dtype=np.float32)

        Xtr_seq = to_seq(Xtr_fused)
        Xval_seq = to_seq(Xval_fused)
        Xte_seq = to_seq(Xte_fused)
        seq_len = len(group_order)
        input_dim = max_cols
        return Xtr_seq, Xval_seq, Xte_seq, seq_len, input_dim, group_order
    else:
        Xtr_seq = Xtr_fused[:, None, :].astype(np.float32)
        Xval_seq = Xval_fused[:, None, :].astype(np.float32)
        Xte_seq = Xte_fused[:, None, :].astype(np.float32)
        return Xtr_seq, Xval_seq, Xte_seq, 1, Xtr_fused.shape[1], feat_names



def build_gcn_features_outer_anchor(X_tr, X_val, X_te, feat_names, model, device, k=10, a=0.1,
                                    grouped=True, mapping=None):

    X_outer = np.vstack([X_tr, X_val]).astype(np.float32)

    E_outer = embed(model, X_outer, device)
    E_te = embed(model, X_te, device)

    S_outer = np.dot(E_outer, E_outer.T)
    S_te = np.dot(E_te, E_outer.T)

    tr_knn = knn_train(S_outer, k)
    tr_norm = l1_norm(tr_knn)
    A_outer, A_tilde_outer, deg_outer = norm_adj_train(tr_norm, a)

    X_outer_s = model._x_scaler.transform(X_outer).astype(np.float32)
    X_te_s = model._x_scaler.transform(X_te).astype(np.float32)

    n_outer = X_outer_s.shape[0]
    n_te = X_te_s.shape[0]

    Xouter_fused = np.dot(A_tilde_outer.astype(np.float32), X_outer_s)

    tr_block_te = np.hstack([A_outer.astype(np.float32), np.zeros((n_outer, n_te), dtype=np.float32)])

    te_block = knn_val(S_te, n_te, n_outer + n_te, k).astype(np.float32)
    full_A_te = np.vstack([tr_block_te, te_block])

    A_te_outer = norm_adj_val(full_A_te, deg_outer, a)
    X_stack_te = np.vstack([X_outer_s, X_te_s]).astype(np.float32)
    Xte_fused = np.dot(A_te_outer, X_stack_te)[-n_te:]

    if grouped:
        if mapping is None:
            raise ValueError("Grouped=True need descriptors mapping JSON。")
        max_cols = max(len(v) for v in mapping.values())
        group_order = list(mapping.keys())
        name2idx = {nm: i for i, nm in enumerate(feat_names)}

        def to_seq(mat):
            rows = []
            for r in range(mat.shape[0]):
                seq = []
                for g in group_order:
                    vals = [float(mat[r, name2idx[c]]) if c in name2idx else 0.0 for c in mapping.get(g, [])]
                    while len(vals) < max_cols:
                        vals.append(0.0)
                    seq.append(vals)
                rows.append(seq)
            return np.array(rows, dtype=np.float32)

        Xouter_seq = to_seq(Xouter_fused)
        Xte_seq = to_seq(Xte_fused)
        seq_len = len(group_order)
        input_dim = max_cols
        return Xouter_seq, Xte_seq, seq_len, input_dim, group_order
    else:
        Xouter_seq = Xouter_fused[:, None, :].astype(np.float32)
        Xte_seq = Xte_fused[:, None, :].astype(np.float32)
        return Xouter_seq, Xte_seq, 1, Xouter_fused.shape[1], feat_names






def train_transformer_with_es(Xtr_seq, y_tr_norm, Xval_seq, y_val_orig,
                              seq_len, input_dim, device,
                              num_heads, num_layers, dim_feedforward,
                              hidden1, hidden2, hidden3, dropout,
                              lr, weight_decay, batch_size,
                              max_epochs, patience,
                              y_scaler=None,
                              grad_clip_norm=1.0):
    tr_loader = DataLoader(
        TensorDataset(to_tensor(Xtr_seq), to_tensor(y_tr_norm.reshape(-1, 1))),
        batch_size=batch_size, shuffle=True
    )

    model = SimpleTransformerRegressor(
        input_dim=input_dim,
        seq_length=seq_len,
        dim_feedforward=dim_feedforward,
        num_heads=num_heads,
        num_layers=num_layers,
        hidden_dim_1=hidden1,
        hidden_dim_2=hidden2,
        hidden_dim_3=hidden3,
        dropout_rate=dropout,
        output_dim=1,
    ).to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    mse_loss = nn.MSELoss()

    best_state = None
    best_r2 = -1e9
    best_epoch = -1
    epochs_no_improve = 0

    for epoch in range(max_epochs):
        model.train()
        for xb, yb in tr_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optim.zero_grad()
            out = model(xb)
            loss = mse_loss(out.float(), yb.float())
            loss.backward()
            if grad_clip_norm is not None and grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            optim.step()

        model.eval()
        with torch.no_grad():
            yhat_val_norm = model(to_tensor(Xval_seq).to(device)).cpu().numpy().ravel()
            yhat_val = inv_if_scaler(yhat_val_norm, y_scaler)
            r2 = float(r2_score(y_val_orig, yhat_val))

        if r2 > best_r2:
            best_r2 = r2
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    return best_state, best_r2, best_epoch





def train_one_fold_by_indices(sheet_name, X, y, cats, feat_names, row_ids,
                              outer_train_idx, outer_test_idx, inner_train_idx, inner_val_idx,
                              descriptors_mapping, args, device, fold_dir, stand_properties):
    ensure_dir(fold_dir)


    idx2pos = {int(rid): pos for pos, rid in enumerate(row_ids.tolist())}
    tr_pos = np.array([idx2pos[i] for i in inner_train_idx], dtype=int)
    val_pos = np.array([idx2pos[i] for i in inner_val_idx], dtype=int)
    te_pos = np.array([idx2pos[i] for i in outer_test_idx], dtype=int)
    X_tr, y_tr, c_tr = X[tr_pos], y[tr_pos], cats[tr_pos]
    X_val, y_val, c_val = X[val_pos], y[val_pos], cats[val_pos]
    X_te, y_te, c_te = X[te_pos], y[te_pos], cats[te_pos]


    print("Start the first stage of training......")

    embed_model = train_embedding_stage(
        X_tr, y_tr, X_val, y_val,
        input_dim=X.shape[1], device=device,
        max_epochs=args.epochs_embed,
        patience=50,
        batch_size=args.bs_embed,
        lr=args.lr_embed, weight_decay=args.wd_embed, margin=args.margin,
    )

    torch.save(embed_model, os.path.join(fold_dir, 'embedding_best.pth'))


    Xtr_seq, Xval_seq, Xte_seq, seq_len, input_dim, group_order = build_gcn_features(
        X_tr, X_val, X_te, feat_names, embed_model, device,
        k=args.k, a=args.a, grouped=True, mapping=descriptors_mapping,
    )

    if args.save_stage15:
        save_path = os.path.join(fold_dir, 'stage15_fused_grouped.xlsx')
        with pd.ExcelWriter(save_path, engine='openpyxl', mode='w') as writer:
            save_stage15_to_xlsx(writer, 'train', c_tr, y_tr, Xtr_seq, group_order)
            save_stage15_to_xlsx(writer, 'val', c_val, y_val, Xval_seq, group_order)
            save_stage15_to_xlsx(writer, 'test', c_te, y_te, Xte_seq, group_order)

    print(f"Start the second stage of training......")


    if sheet_name not in stand_properties:
        y_scaler_tr = LogTargetScaler(base=10, eps=None, with_std=True)
        y_scaler_tr.fit(y_tr)
        y_tr_norm = y_scaler_tr.transform(y_tr)
    else:
        y_scaler_tr = StandardScaler()
        y_tr_norm = y_scaler_tr.fit_transform(y_tr.reshape(-1, 1)).ravel()
    best_states = {}
    trial_best_epoch = {}
    best_scalers = {}

    def objective(trial):
        num_layers = trial.suggest_int('num_layers', 1, 4)

        # Set the moleculeNet dataset to [1, 2, 3, 6].
        num_heads = trial.suggest_categorical('num_heads', [1, 2])

        dim_feedforward = trial.suggest_categorical('dim_feedforward', [256, 512, 1024])
        hidden1 = trial.suggest_categorical('hidden1', [64, 128, 256])
        hidden2 = trial.suggest_categorical('hidden2', [512, 1024, 2048])
        hidden3 = trial.suggest_categorical('hidden3', [64, 128, 256])
        dropout = trial.suggest_float('dropout', 0.1, 0.5)
        lr = trial.suggest_float('lr', 1e-4, 3e-3, log=True)
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)

        best_state, best_r2, best_epoch = train_transformer_with_es(
            Xtr_seq, y_tr_norm, Xval_seq, y_val,
            seq_len, input_dim, device,
            num_heads, num_layers, dim_feedforward,
            hidden1, hidden2, hidden3, dropout,
            lr, weight_decay, args.bs_tr,
            args.epochs_tr, args.patience,
            y_scaler=y_scaler_tr,
            grad_clip_norm=1.0,
        )
        best_states[trial.number] = best_state
        trial_best_epoch[trial.number] = int(best_epoch)
        best_scalers[trial.number] = y_scaler_tr
        return best_r2

    study = optuna.create_study(direction='maximize', study_name=sheet_name,
                                sampler=optuna.samplers.TPESampler(seed=args.seed),
                                pruner=optuna.pruners.MedianPruner(n_warmup_steps=10)
                                )
    study.optimize(objective, n_trials=args.trials, show_progress_bar=False)

    best_trial = study.best_trial
    best_params = best_trial.params
    best_epoch = trial_best_epoch.get(best_trial.number, -1)
    best_state = best_states.get(best_trial.number)

    if best_state is None:
        best_state, _, best_epoch = train_transformer_with_es(
            Xtr_seq, y_tr_norm, Xval_seq, y_val,
            seq_len, input_dim, device,
            best_params['num_heads'], best_params['num_layers'], best_params['dim_feedforward'],
            best_params['hidden1'], best_params['hidden2'], best_params['hidden3'],
            best_params['dropout'],
            best_params['lr'], best_params['weight_decay'], args.bs_tr,
            args.epochs_tr, args.patience,
            y_scaler=y_scaler_tr,
            grad_clip_norm=1.0,
        )



    Xouter_seq, Xte_seq_outer, seq_len2, input_dim2, group_order2 = build_gcn_features_outer_anchor(
        X_tr, X_val, X_te, feat_names, embed_model, device,
        k=args.k, a=args.a, grouped=True, mapping=descriptors_mapping
    )

    y_full = np.concatenate([y_tr, y_val], axis=0)
    if sheet_name not in stand_properties:
        final_scaler = LogTargetScaler(base=10, eps=None, with_std=True)
        final_scaler.fit(y_full)
        y_full_norm = final_scaler.transform(y_full)
    else:
        final_scaler = StandardScaler()
        y_full_norm = final_scaler.fit_transform(y_full.reshape(-1, 1)).ravel()


    final_model = SimpleTransformerRegressor(
        input_dim=input_dim2,
        seq_length=seq_len2,
        dim_feedforward=best_params['dim_feedforward'],
        num_heads=best_params['num_heads'],
        num_layers=best_params['num_layers'],
        hidden_dim_1=best_params['hidden1'],
        hidden_dim_2=best_params['hidden2'],
        hidden_dim_3=best_params['hidden3'],
        dropout_rate=best_params['dropout'],
        output_dim=1,
    ).to(device)

    fixed_epochs = int(best_epoch) if int(best_epoch) > 0 else max(20, int(args.epochs_tr * 0.5))

    tr_loader_full = DataLoader(
        TensorDataset(to_tensor(Xouter_seq), to_tensor(y_full_norm.reshape(-1, 1))),
        batch_size=args.bs_tr, shuffle=True
    )
    optim = torch.optim.Adam(final_model.parameters(), lr=best_params['lr'], weight_decay=best_params['weight_decay'])
    final_model.train()
    for epoch in range(fixed_epochs):
        for xb, yb in tr_loader_full:
            xb = xb.to(device);
            yb = yb.to(device)
            optim.zero_grad()
            out = final_model(xb)
            loss = mse_loss(out.float(), yb.float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(final_model.parameters(), max_norm=1.0)
            optim.step()


    if args.save_stage15:
        save_path_outer = os.path.join(fold_dir, 'stage15_fused_grouped_outer.xlsx')
        with pd.ExcelWriter(save_path_outer, engine='openpyxl', mode='w') as writer:
            c_outer = np.concatenate([c_tr, c_val], axis=0)
            y_outer = y_full
            save_stage15_to_xlsx(writer, 'train', c_outer, y_outer, Xouter_seq, group_order2)
            save_stage15_to_xlsx(writer, 'test', c_te, y_te, Xte_seq_outer, group_order2)


    final_model.eval()
    with torch.no_grad():
        yhat_te_norm = final_model(to_tensor(Xte_seq_outer).to(device)).cpu().numpy().ravel()
        yhat_te = inv_if_scaler(yhat_te_norm, final_scaler)
        r2 = float(r2_score(y_te, yhat_te))

    torch.save(final_model, os.path.join(fold_dir, 'transformer_best.pth'))
    torch.save(final_scaler, os.path.join(fold_dir, 'y_scaler.pth'))

    print(
        f"Fold (dir={os.path.basename(fold_dir)}): Test R2 = {round(r2, 4)}  (seq_len={seq_len}, input_dim={input_dim})")

    test_pred = pd.DataFrame({
        "row_id": np.array(outer_test_idx, dtype=int),
        "category": c_te,
        "y_true": y_te.astype(float),
        "y_pred": yhat_te.astype(float),
    })
    test_pred.to_csv(os.path.join(fold_dir, 'test_predictions.csv'), index=False)


    split_info = {
        "Test R2": r2,
        "sheet": sheet_name,
        "seed": int(args.seed),
        "stratified_by": "category+log(y)",
        "k": int(args.k),
        "a": float(args.a),
        "mapping_path": os.path.abspath(args.mapping),
        "final_anchor": "outer_train",
        "seq_len": int(seq_len),
        "input_dim": int(input_dim),
        "outer_train_idx": [int(x) for x in outer_train_idx],
        "outer_test_idx": [int(x) for x in outer_test_idx],
        "inner_train_idx": [int(x) for x in inner_train_idx],
        "inner_val_idx": [int(x) for x in inner_val_idx],
        "best_trial_number": int(best_trial.number),
        "best_epoch": int(best_epoch),
        "best_params": best_params,
        "files": {
            "embedding": os.path.join(fold_dir, 'embedding_best.pth'),
            "transformer": os.path.join(fold_dir, 'transformer_best.pth'),
            "y_scaler": os.path.join(fold_dir, 'y_scaler.pth'),
            "pred_csv": os.path.join(fold_dir, 'test_predictions.csv')
        }
    }
    with open(os.path.join(fold_dir, 'split.json'), 'w', encoding='utf-8') as fp:
        json.dump(split_info, fp, ensure_ascii=False, indent=2)

    return r2
