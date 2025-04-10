import pandas as pd
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.metrics.pairwise import cosine_similarity
import ast
from common.parse_args import args




def splitGroupGCNData(file_path, sheet_name):
    # 前五个是sheet6，后面13个是sheet3
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    data_length = df.shape[0]
    labels = df.iloc[:, 0].values
    labels = labels.reshape(-1, 1)
    data = df.iloc[:, 1:].values

    # 将每个特征值从str转到array
    def parse_feature_values(data):
        # 对每个特征值应用 ast.literal_eval，将字符串转换为列表
        return [np.array(ast.literal_eval(item)).astype('float32') for item in data]
    # 提取训练集和测试集的特征和标签
    data = np.array([parse_feature_values(sample) for sample in data])

    split = int(data_length * 0.8)
    x_train = data[:split, :].astype('float32')
    x_test = data[split:, :].astype('float32')
    y_train = labels[:split].astype('float32')
    y_test = labels[split:].astype('float32')

    return x_train, x_test, y_train, y_test








def getSimilarityMmatrix(file_path):
    df = pd.read_excel(file_path, sheet_name='Sheet2')
    # 去除没有类型的数据
    df = df.dropna(subset=['Class'])
    # 按照类别分组
    grouped = df.groupby(df.columns[0])
    # 初始化空列表
    pre_data_len = len(df.columns)
    train_set = np.empty((0, pre_data_len))
    test_set = np.empty((0, pre_data_len))
    for name, group in grouped:
        # (group_len, 1290)
        group_data = group.values
        group_length = group_data.shape[0]
        # 打乱顺序
        np.random.seed(42)
        np.random.shuffle(group_data)
        # 划分训练集和测试集
        split = int(group_length * 0.8)
        train_data = group_data[:split, :]
        test_data = group_data[split:, :]
        # 添加到对应集合
        train_set = np.concatenate((train_set, train_data), axis=0)
        test_set = np.concatenate((test_set, test_data), axis=0)
    x_train = train_set[:, 2:].astype(np.float32)
    x_test = test_set[:, 2:].astype(np.float32)

    X = np.vstack((x_train, x_test))
    similarity_matrix = np.dot(X, X.T)
    return similarity_matrix


# 单独获取相似矩阵和Y值列
def getAdjacMatrix(file_path):
    x_train, x_test, y_train, y_test = splitTrainTest(file_path)
    X = np.vstack((x_train, x_test))
    labels = np.vstack((y_train, y_test))

    # 得到相似性矩阵
    similarity_matrix = getSimilarityMmatrix(file_path)
    return X, similarity_matrix, labels



# 对相似矩阵做归一化操作以及划分测试集和训练集
def handle_NormAdjacMatrix(pre_matrix, adjac_matrix, labels, k, a):
    # 保留前k大的数
    similarity_matrix = reserveKNum(adjac_matrix, k)
    # 归一化
    A = matrixNorm(similarity_matrix, a)
    gcn_matrix = np.dot(A, pre_matrix)

    length = len(pre_matrix)
    # 划分训练集和测试集
    split = int(length * 0.8)
    x_train = gcn_matrix[:split, :].astype(np.float32)
    x_test = gcn_matrix[split:, :].astype(np.float32)
    y_train = labels[:split, :].astype(np.float32)
    y_test = labels[split:, :].astype(np.float32)

    x_train = normalFunction(x_train)
    x_test = normalFunction(x_test)

    return x_train, x_test, y_train, y_test



# 划分描述符的GCN数据
def splitGCNDescriptors(file_path, k, a):
    x_train, x_test, y_train, y_test = splitTrainTest(file_path)
    length = len(x_train)
    features = np.vstack((x_train, x_test))

    norm_matrix = matrixFromGCN(features, k, a)

    # 划分训练集和测试集
    x_train = norm_matrix[:length, :]
    x_test = norm_matrix[length:, :]

    x_train = normalFunction(x_train)
    x_test = normalFunction(x_test)
    return x_train, x_test, y_train, y_test




# 划分transformer的描述符数据
def splitTrainTest(file_path):
    df = pd.read_excel(file_path, sheet_name='Sheet5')
    # 去除没有类型的数据
    df = df.dropna(subset=['Class'])
    # 按照类别分组
    grouped = df.groupby(df.columns[0])
    # 初始化空列表
    pre_data_len = len(df.columns)
    train_set = np.empty((0, pre_data_len))
    test_set = np.empty((0, pre_data_len))
    for name, group in grouped:
        # group_data (group_len, 1290)
        group_data = group.values
        group_length = group_data.shape[0]

        # 打乱顺序
        np.random.seed(42)
        np.random.shuffle(group_data)

        # 划分训练集和测试集
        split = int(group_length * 0.8)
        train_data = group_data[:split, :]
        test_data = group_data[split:, :]

        # 添加到对应集合
        train_set = np.concatenate((train_set, train_data), axis=0)
        test_set = np.concatenate((test_set, test_data), axis=0)

    # 使用 ast.literal_eval() 来将字符串转换为列表
    def parse_features(data):
        # print(data.shape)
        return [np.array(ast.literal_eval(item)) for item in data]

    # 提取训练集和测试集的特征和标签
    x_train = np.array([parse_features(sample[2:]) for sample in train_set])
    x_test = np.array([parse_features(sample[2:]) for sample in test_set])
    y_train = np.array([sample[1] for sample in train_set])  # 标签列
    y_test = np.array([sample[1] for sample in test_set])    # 标签列

    return x_train, x_test, y_train, y_test


# 划分GCN的指纹数据
def splitFPdata(file_path, smiles_column, label_column, data_dim, k, a):
    data, labels = getFpdata(file_path, smiles_column, label_column, data_dim, k, a)
    # 调整 labels 的形状以匹配 X 的行数
    labels = labels.reshape(-1, 1)
    results = np.hstack((labels, data))
    length = len(results)
    np.random.seed(42)
    np.random.shuffle(results)
    # 划分训练集和测试集
    split = int(length * 0.8)
    train_data = results[:split, :]
    test_data = results[split:, :]
    x_train = train_data[:, 1:].astype(np.float32)
    x_test = test_data[:, 1:].astype(np.float32)
    y_train = train_data[:, 0:1].astype(np.float32)
    y_test = test_data[:, 0:1].astype(np.float32)

    x_train = normalFunction(x_train)
    x_test = normalFunction(x_test)
    return x_train, x_test, y_train, y_test





# 获取GCN摩根指纹数据并进行归一化， 从unique文件里读取
def getFpdata(file_path, smiles_column, label_column, data_dim, k, a):
    df = pd.read_excel(file_path, sheet_name='Sheet1')
    smiles = df[smiles_column]
    labels = df[label_column].values
    fp_arr = []
    for smile in smiles:
        # 通过smiles获取分子结构对象
        mol = Chem.MolFromSmiles(smile)
        # 获取分子半径为 2 的 Morgan 指纹，nBits 参数设置指纹的位数，通常这个值是 2048 或其他，由变量 fp_dim 指定。
        # useChirality=True 表示在生成指纹时考虑分子的手性
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=int(data_dim), useChirality=True)
        # 获取该数组中值为1的所有索引
        onbits = list(fp.GetOnBits())
        # 生成一个同等大小的全0数组
        arr = np.zeros(fp.GetNumBits())
        # 将该数组对应索引位置置1，得到特征数组
        arr[onbits] = 1
        fp_arr.append(arr)

    result = matrixFromGCN(fp_arr, k, a)
    saveToExcel(result, 'D:\\code\\PyCharm_WorkSpace\\ai4fuel_transformer_2\\data\\hov_fp.xlsx')
    return result, labels


# 归一化GCN的邻接矩阵
def matrixFromGCN(matrix, k, a):
    X = np.array(matrix)
    # 得到相似性矩阵
    similarity_matrix = np.dot(X, X.T)

    # 保留前k大的数
    similarity_matrix = reserveKNum(similarity_matrix, k)
    # 归一化
    norm_similarity_matrix = matrixNorm(similarity_matrix)
    A = norm_similarity_matrix * a + np.eye(similarity_matrix.shape[0])
    result = np.dot(A, matrix)
    return result


def saveToExcel(matrix, path):
    df = pd.DataFrame(matrix)
    df.to_excel(path, index=False)  # index=False表示不保存行索引到文件

# 矩阵每一行保留前K个最大的数
def reserveKNum(matrix, k):

    # 获取每行前 k 个最大值的索引
    sorted_indices = np.argsort(matrix, axis=1)[:, ::-1]  # 每行从大到小排序的索引
    top_k_indices = sorted_indices[:, :k]  # 取前 k 个索引
    # 创建一个零矩阵
    result = np.zeros_like(matrix)
    # 将前 k 个最大值的位置赋值
    rows = np.arange(matrix.shape[0])[:, None]  # 每行的索引
    result[rows, top_k_indices] = matrix[rows, top_k_indices]

    return result

# 邻接矩阵余弦相似度归一化
def matrixNorm(matrix, a):
    # 计算每行向量的 L1 范数
    l1_norms = np.sum(np.abs(matrix), axis=1)
    # 创建一个行 L1 范数的广播数组
    l1_norms_matrix = l1_norms[:, np.newaxis]
    # 执行元素级的除法，避免除以零
    l1_normalized_matrix = matrix / (l1_norms_matrix + (l1_norms_matrix == 0))
    # 聚合操作
    A = l1_normalized_matrix * a + np.eye(l1_normalized_matrix.shape[0])

    # 计算度矩阵 D
    degree_matrix = np.sum(A, axis=1)
    # 防止度为零，添加一个小常数
    D_inv_sqrt = np.diag(1.0 / np.sqrt(degree_matrix + 1e-10))
    # 归一化处理后的邻接矩阵
    result = D_inv_sqrt @ A @ D_inv_sqrt
    return result


def normalFunction(array):
    # X标准化
    scaler_standard = StandardScaler()
    result = scaler_standard.fit_transform(array)
    return result


def handle_PCA(array):
    pca = PCA(n_components=args.input_dim, svd_solver='randomized')
    result = pca.fit_transform(array)
    return result


if __name__ == '__main__':
    # file_path = 'D:\\code\\PyCharm_WorkSpace\\ai4fuel\\data\\HOV.xlsx'
    # smiles_column = 'smiles_list'
    # label_column = 'HOV'
    # # splitFPdata(file_path, smiles_column, label_column, data_dim)
    # x_train, x_test, y_train, y_test = splitTrainTest(file_path)
    # print(x_train.shape, x_test.shape)
    # for mol in x_train:
    #     for descriptor in mol:
    #         print(type(descriptor))
    # print(y_train.shape, y_test.shape)

    # X, similarity_matrix, labels = getAdjacMatrix(file_path)
    # print(X.shape, similarity_matrix.shape, labels.shape)
    file_path = "/data/all/DCN.xlsx"
    x_train, x_test, y_train, y_test = splitGroupGCNData(file_path, 'Sheet3')


