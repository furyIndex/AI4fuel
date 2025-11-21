
import os
import json
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler


def load_sheet(xlsx_path, sheet_name):
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name)
    if df.shape[1] < 3:
        raise ValueError("At least three columns are needed：category, label, features...")
    cats = df.iloc[:, 0].values
    y = df.iloc[:, 1].values.astype(np.float32)
    X = df.iloc[:, 2:].values.astype(np.float32)
    feat_names = [str(c) for c in df.columns[2:]]
    row_ids = np.arange(len(df), dtype=np.int64)
    return X, y, cats, feat_names, row_ids


def embed(model, X, device):
    x_s = model._x_scaler.transform(X)
    with torch.no_grad():
        e = model(torch.tensor(x_s, dtype=torch.float32).to(device)).cpu().numpy()
    return e


def knn_train(similarity_matrix, k):
    n = similarity_matrix.shape[0]
    knn_matrix = np.zeros_like(similarity_matrix)
    for i in range(n):
        top_k_indices = np.argpartition(similarity_matrix[i], -k)[-k:]
        knn_matrix[i, top_k_indices] = similarity_matrix[i, top_k_indices]
    return knn_matrix


def l1_norm(adj_matrix):
    row_sum = np.sum(adj_matrix, axis=1)
    row_sum[row_sum == 0] = 1
    normalized_matrix = adj_matrix / row_sum[:, np.newaxis]
    return normalized_matrix


def norm_adj_train(adj_matrix, alpha):

    n = adj_matrix.shape[0]
    A_tilde = adj_matrix + alpha * np.eye(n)
    deg = np.sum(A_tilde, axis=1)
    deg_sqrt_inv = np.power(deg, -0.5)
    deg_sqrt_inv[deg_sqrt_inv == float('inf')] = 0
    D_sqrt_inv = np.diag(deg_sqrt_inv)
    A_norm = D_sqrt_inv @ A_tilde @ D_sqrt_inv
    return A_tilde, A_norm, deg


def knn_val(test_similarity_matrix, test_length, full_length, k):
    test_topk = np.argpartition(test_similarity_matrix, -k, axis=1)[:, -k:]
    test_adj = np.zeros((test_length, full_length))
    row_indices = np.arange(test_length)[:, None]
    test_adj[row_indices, test_topk] = test_similarity_matrix[row_indices, test_topk]
    test_block = l1_norm(test_adj)
    return test_block


def norm_adj_val(full_adj, train_degree, a):

    N = len(train_degree)


    full_adj[N:] = full_adj[N:] * a
    full_adj[N:, N:] = full_adj[N:, N:] + np.eye(full_adj.shape[0]-N)


    degree = np.zeros(full_adj.shape[0])
    degree[:N] = train_degree


    test_degree = np.sum(full_adj[N:], axis=1)
    degree[N:] = test_degree


    degree_sqrt_inv = np.diag(degree ** -0.5)
    return degree_sqrt_inv @ full_adj @ degree_sqrt_inv


def build_gcn_features_for_external(X_train, X_external, feat_names, embed_model, device, k=10, a=0.1):

    E_train = embed(embed_model, X_train, device)
    E_external = embed(embed_model, X_external, device)


    S_train = np.dot(E_train, E_train.T)
    S_external = np.dot(E_external, E_train.T)


    tr_knn = knn_train(S_train, k)
    tr_norm = l1_norm(tr_knn)
    A_tilde_train, A_train_norm, deg_train = norm_adj_train(tr_norm, a)


    X_train_s = embed_model._x_scaler.transform(X_train).astype(np.float32)
    X_external_s = embed_model._x_scaler.transform(X_external).astype(np.float32)

    n_train = X_train_s.shape[0]
    n_external = X_external_s.shape[0]


    X_train_fused = np.dot(A_train_norm.astype(np.float32), X_train_s)


    tr_block_external = np.hstack([
        A_tilde_train.astype(np.float32),
        np.zeros((n_train, n_external), dtype=np.float32)
    ])
    external_block = knn_val(S_external, n_external, n_train + n_external, k).astype(np.float32)
    full_A_external = np.vstack([tr_block_external, external_block])

    A_external = norm_adj_val(full_A_external, deg_train, a)
    X_stack_external = np.vstack([X_train_s, X_external_s]).astype(np.float32)
    X_external_fused = np.dot(A_external, X_stack_external)[-n_external:]

    return X_train_fused, X_external_fused


def save_fused_features_to_xlsx(output_path, cats_test, y_test, X_test_original, X_test_fused, feat_names):

    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:

        original_data = {
            "category": list(cats_test),
            "label": [float(v) for v in y_test]
        }

        for i, feat_name in enumerate(feat_names):
            original_data[feat_name] = [float(X_test_original[j, i]) for j in range(X_test_original.shape[0])]

        df_original = pd.DataFrame(original_data)
        df_original.to_excel(writer, sheet_name='original', index=False)


        fused_data = {
            "category": list(cats_test),
            "label": [float(v) for v in y_test]
        }

        for i, feat_name in enumerate(feat_names):
            fused_data[feat_name] = [float(X_test_fused[j, i]) for j in range(X_test_fused.shape[0])]

        df_fused = pd.DataFrame(fused_data)
        df_fused.to_excel(writer, sheet_name='fused', index=False)





def load_embedding_model(embed_model_path, device):

    model = torch.load(embed_model_path, map_location=device, weights_only=False)
    return model




def process_external_test_set_from_split(fold_dir, descriptors_all_path, train_output_path, test_output_path, all_output_path):
    split_path = os.path.join(fold_dir, 'split.json')
    with open(split_path, 'r', encoding='utf-8') as f:
        split_info = json.load(f)


    sheet_name = split_info['sheet']
    X_all, y_all, cats_all, feat_names, row_ids = load_sheet(descriptors_all_path, sheet_name)


    idx2pos = {int(rid): pos for pos, rid in enumerate(row_ids.tolist())}


    outer_train_idx = split_info['outer_train_idx']
    outer_test_idx = split_info['outer_test_idx']


    train_pos = np.array([idx2pos[i] for i in outer_train_idx], dtype=int)
    test_pos = np.array([idx2pos[i] for i in outer_test_idx], dtype=int)

    X_train = X_all[train_pos]
    y_train = y_all[train_pos]
    cats_train = cats_all[train_pos]

    X_test = X_all[test_pos]
    y_test = y_all[test_pos]
    cats_test = cats_all[test_pos]


    embed_model_path = os.path.join(fold_dir, 'embedding_best.pth')


    device = torch.device('cuda')
    embed_model = load_embedding_model(embed_model_path, device)


    k = split_info.get('k', 10)
    a = split_info.get('a', 0.1)



    X_train_fused, X_test_fused = build_gcn_features_for_external(
        X_train, X_test, feat_names, embed_model, device, k=k, a=a
    )

    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)
    cats_train = cats_train.reshape(-1, 1)
    cats_test = cats_test.reshape(-1, 1)

    X_all_fused = np.vstack([X_train_fused, X_test_fused])
    X_all = np.vstack([X_train, X_test])
    y_all = np.vstack([y_train, y_test])
    cats_all = np.vstack([cats_train, cats_test])

    cats_all = cats_all.reshape(-1)
    cats_train = cats_train.reshape(-1)
    cats_test = cats_test.reshape(-1)



    save_fused_features_to_xlsx(
        test_output_path,
        cats_test,
        y_test,
        X_test,
        X_test_fused,
        feat_names
    )



    save_fused_features_to_xlsx(
        train_output_path,
        cats_train,
        y_train,
        X_train,
        X_train_fused,
        feat_names
    )




    save_fused_features_to_xlsx(
        all_output_path,
        cats_all,
        y_all,
        X_all,
        X_all_fused,
        feat_names
    )


def merge_fuel_types(input_file, sheet_name, output_file, new_sheet_name):

    merge_rules = {
        'n-Alkanes': 'HC',
        'iso-Alkanes': 'HC',
        'Alkenes': 'HC',
        'Alkynes': 'HC',
        'Cycloalkanes': 'CHC',
        'Cyclic alkenes': 'CHC',
        'Bicycloalkanes': 'CHC',
        'Aromatics': 'AROM',
        'Terpenes': 'AROM',
        'Alcohols': 'ALET',
        'Acyclic ethers': 'ALET',
        'Cyclic ethers': 'ALET',
        'Other cyclic ethers': 'ALET',
        'Carbonate ester': 'ES',
        'Saturated esters': 'ES',
        'Unsaturated esters': 'ES',
        'Aldehydes': 'ALKE',
        'Cyclic ketone': 'ALKE',
        'Ketones': 'ALKE',
        'Carboxylic acids': 'Ac-X',
        'Carboxylic anhydride': 'Ac-X',
        'Amides': 'Ac-X',
        'Polyfunctionals': 'PolyF',
        'Peroxide': 'POX',
        'Furans': 'EPOX',
        'Phenols': 'EPOX'
    }

    df = pd.read_excel(input_file, sheet_name=sheet_name)

    fuel_types = df.iloc[:, 0]

    merged_fuel_types = fuel_types.map(merge_rules).fillna(fuel_types)

    new_sheet_data = pd.DataFrame({df.columns[0]: merged_fuel_types})

    with pd.ExcelWriter(input_file, engine='openpyxl', mode='a', if_sheet_exists='new') as writer:

        new_sheet_data.to_excel(writer, sheet_name=new_sheet_name, index=False)

    print(f"Successfully saved to new worksheet: '{new_sheet_name}' (Original file: {input_file})")









if __name__ == '__main__':

    fold_dir = "../train_kfold/lg_transformer_kfold_runs/FE/fold1"
    descriptors_all_path = "../data/revision/descriptors_all.xlsx"
    train_output_path = "train_fused_features_FE1.xlsx"
    test_output_path = "test_fused_features_FE1.xlsx"
    all_output_path = "fused_features_FE1.xlsx"

    process_external_test_set_from_split(
        fold_dir, descriptors_all_path, train_output_path, test_output_path, all_output_path
    )


    merge_fuel_types(train_output_path, "fused", train_output_path, "category")
    merge_fuel_types(test_output_path, "fused", test_output_path, "category")
    merge_fuel_types(all_output_path, "fused", test_output_path, "category")

    print("种类合并完成")
