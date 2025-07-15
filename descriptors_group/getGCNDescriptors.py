import json
import os
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from common.parse_args import args


device = torch.device(args.device)




def save_gcn_data(file_path, sheet_name, k, a):

    x_train, x_test, labels, descriptor_name = splitTrainTest(file_path)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    x_full = np.vstack((x_train, x_test))
    train_length = len(x_train)
    test_length = len(x_test)
    full_length = len(x_full)


    train_similarity_matrix = np.dot(x_train, x_train.T)
    test_similarity_matrix = np.dot(x_test, x_train.T)


    train_knn_matrix = knn_train(train_similarity_matrix, k)
    train_norm_matrix = l1_norm(train_knn_matrix)
    train_A, train_degree = norm_adj_train(train_norm_matrix, a)
    train_gcn_matrix = np.dot(train_A, x_train)


    test_block = knn_val(test_similarity_matrix, test_length, full_length, k)


    train_block = np.hstack([train_A, np.zeros((train_length, test_length))])
    full_adj = np.vstack([train_block, test_block])


    full_A = norm_adj_val(full_adj, train_degree, a)
    full_gcn_matrix = np.dot(full_A, x_full)
    test_gcn_matrix = full_gcn_matrix[-test_length:]


    gcn_matrix = np.vstack((train_gcn_matrix, test_gcn_matrix))
    pre_group_df = pd.DataFrame(gcn_matrix, columns=descriptor_name)


    max_columns = max(len(descriptors) for descriptors in descriptorsMapping.values())
    result_df  = pd.DataFrame()


    for type, descriptors in descriptorsMapping.items():
        exist_df = [descriptor for descriptor in descriptors if descriptor in pre_group_df.columns]
        if exist_df:
            group_df = pre_group_df[exist_df]
            padded_values = group_df.apply(lambda x: x.tolist() + [0] * (max_columns - len(x)), axis=1)
            result_df[type] = padded_values
    result_df.insert(loc=0, column='value', value=labels)

    with pd.ExcelWriter(file_path, engine='openpyxl', mode='a') as writer:
        result_df.to_excel(writer, sheet_name=sheet_name, index=False)




def splitTrainTest(file_path):
    df = pd.read_excel(file_path, sheet_name='Sheet2')
    all_columns = df.columns
    descriptor_name = all_columns[2:].tolist()

    grouped = df.groupby(df.columns[0])

    pre_data_len = len(df.columns)
    train_set = np.empty((0, pre_data_len))
    test_set = np.empty((0, pre_data_len))
    for name, group in grouped:
        group_data = group.values
        group_length = group_data.shape[0]

        np.random.seed(42)
        np.random.shuffle(group_data)

        split = int(group_length * 0.8)
        train_data = group_data[:split, :]
        test_data = group_data[split:, :]

        train_set = np.concatenate((train_set, train_data), axis=0)
        test_set = np.concatenate((test_set, test_data), axis=0)
    full_data = np.vstack((train_set, test_set))
    split = int(full_data.shape[0] * 0.8)
    x_train = full_data[:split, 2:].astype(np.float32)
    x_test = full_data[split:, 2:].astype(np.float32)
    labels = full_data[:, 1:2].astype(np.float32)

    return x_train, x_test, labels, descriptor_name



def knn_train(X, k):

    num_nodes = X.shape[0]
    np.fill_diagonal(X, 0)
    adj = np.zeros_like(X)


    topk_indices = np.argpartition(X, -k, axis=1)[:, -k:]
    row_indices = np.arange(num_nodes)[:, np.newaxis]
    adj[row_indices, topk_indices] = X[row_indices, topk_indices]

    return adj



def knn_val(test_similarity_matrix, test_length, full_length, k):
    test_topk = np.argpartition(test_similarity_matrix, -k, axis=1)[:, -k:]
    test_adj = np.zeros((test_length, full_length))
    row_indices = np.arange(test_length)[:, None]
    test_adj[row_indices, test_topk] = test_similarity_matrix[row_indices, test_topk]
    test_block = l1_norm(test_adj)
    return test_block



def l1_norm(matrix):
    l1_norms = np.sum(np.abs(matrix), axis=1)
    l1_norms_matrix = l1_norms[:, np.newaxis]
    l1_normalized_matrix = matrix / (l1_norms_matrix + (l1_norms_matrix == 0))
    return l1_normalized_matrix



def norm_adj_train(train_matrix, alpha=1.0):

    A = train_matrix * alpha + np.eye(train_matrix.shape[0])
    train_degree = np.sum(A, axis=1)
    D_inv_sqrt = np.diag(train_degree ** -0.5)
    result = D_inv_sqrt @ A @ D_inv_sqrt
    return result, train_degree


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













if __name__ == '__main__':
    with open('./descriptorsMap/descriptorsMapping.json', 'r') as f:
        descriptorsMapping = json.load(f)

    sheet_list = [
        '10_0.1',
        # '10_0.2',
        # '10_0.5',
        # '10_1',
        # '50_0.1',
        # '50_0.2',
        # '50_0.5',
    ]
    for sheet_name in sheet_list:

        k = int(sheet_name.split('_')[0])
        a = float(sheet_name.split('_')[1])
        sheet_name_to_check = "k={}_a={}".format(k, a)
        print("============={}==============".format(sheet_name_to_check))

        data_folder_path = 'D:\\code\\PyCharm_WorkSpace\\ai4fuel\\data\\other'
        info_list = []
        for filename in os.listdir(data_folder_path):

            file_path = os.path.join(data_folder_path, filename)
            print(file_path)

            df = pd.read_excel(file_path, sheet_name=None)
            if sheet_name_to_check in df.keys():
                continue

            save_gcn_data(file_path, sheet_name_to_check, k, a)