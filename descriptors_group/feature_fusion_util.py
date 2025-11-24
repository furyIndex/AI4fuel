import numpy as np


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
    D_inv_sqrt = np.diag(train_degree ** -0.5)  # D^(-1/2)
    A_tilde = D_inv_sqrt @ A @ D_inv_sqrt
    return A, A_tilde, train_degree


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
