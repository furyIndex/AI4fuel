import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import random
from common.parse_args import args


def descriptersFromExcel(path):
    # 读取Excel文件
    df = pd.read_excel(path, sheet_name='Sheet1')
    # 访问列
    arr = np.array(df, dtype=np.float32)
    print(arr.shape)
    return arr



def smilesFromExcel():
    # 读取Excel文件
    df = pd.read_excel('D:\\code\\PyCharm_WorkSpace\\ai4fuel\\data\\HoV_unique.xlsx', sheet_name='Sheet1')
    # 访问列
    smiles_list = df['smiles_list']
    return smiles_list.values

def hoVFromExcel():
    # 读取Excel文件
    df = pd.read_excel('D:\\code\\PyCharm_WorkSpace\\ai4fuel\\data\\HoV_unique.xlsx', sheet_name='Sheet1')
    # 访问列
    targets = df['HOV']
    return np.array(targets.values, dtype=np.float32)

def LHVFromExcel():
    # 读取Excel文件
    df = pd.read_excel('D:\\code\\PyCharm_WorkSpace\\ai4fuel\\data\\LHV_unique.xlsx', sheet_name='Sheet1')
    # 访问列
    targets = df['LHV']
    return np.array(targets.values, dtype=np.float32)

def RONFromExcel():
    # 读取Excel文件
    df = pd.read_excel('D:\\code\\PyCharm_WorkSpace\\ai4fuel\\data\\RON-MON-CN.xlsx', sheet_name='RON')
    # 访问列
    targets = df['RON_Measured']
    return np.array(targets.values, dtype=np.float32)

def MONFromExcel():
    # 读取Excel文件
    df = pd.read_excel('D:\\code\\PyCharm_WorkSpace\\ai4fuel\\data\\RON-MON-CN.xlsx', sheet_name='MON')
    # 访问列
    targets = df['MON_Measured']
    return np.array(targets.values, dtype=np.float32)

def CNFromExcel():
    # 读取Excel文件
    df = pd.read_excel('D:\\code\\PyCharm_WorkSpace\\ai4fuel\\data\\RON-MON-CN.xlsx', sheet_name='CN')
    # 访问列
    targets = df['CN_Measured']
    return np.array(targets.values, dtype=np.float32)

def getMorganFromSmiles(smiles, in_dim):
    # 通过smiles获取分子结构对象
    mol = Chem.MolFromSmiles(smiles)
    # 获取分子半径为 2 的 Morgan 指纹，nBits 参数设置指纹的位数，通常这个值是 2048 或其他，由变量 fp_dim 指定。
    # useChirality=True 表示在生成指纹时考虑分子的手性
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=int(in_dim),useChirality=True)
    # 获取该数组中值为1的所有索引
    onbits = list(fp.GetOnBits())
    # 生成一个同等大小的全0数组
    arr = np.zeros(fp.GetNumBits())
    # 将该数组对应索引位置置1，得到特征数组
    arr[onbits] = 1
    # 转换成（1，2048）的shape
    arr = np.reshape(arr,[arr.shape[0]])
    arr = torch.tensor(arr, dtype=torch.float32)

    # if self.device >= 0:
    #     arr = arr.to(self.device)
    return arr



def getAllData(file_path):
    df = pd.read_excel(file_path, sheet_name='Sheet2')
    print(df.info)
    x_data = np.array(df.iloc[:, 2:]).astype(np.float32)
    y_data = np.array(df.iloc[:, 1:2]).astype(np.float32)
    print(x_data.shape)
    print(y_data.shape)
    return x_data, y_data


def splitTrainTest(file_path):

    df = pd.read_excel(file_path, sheet_name='Sheet2')
    # 先对x进行标准化/归一化
    if args.preHandle_data_normalization == 1:
        df = preHandleDF(df)
    # 按照类别分组
    grouped = df.groupby(df.columns[0])
    # 初始化空列表
    train_set = np.empty((0, args.preHandle_data_dim))
    test_set = np.empty((0, args.preHandle_data_dim))
    for name, group in grouped:
        print(name)
        print(group.values)
        # (group_len, 1290)
        group_data = group.values
        group_length = group_data.shape[0]
        np.random.shuffle(group_data)
        # 划分训练集和测试集
        split = int(group_length * 0.8)
        train_data = group_data[:split, :]
        test_data = group_data[split:, :]

        print("该group中数据总长：")
        print(group_length)
        print("该group中训练数据长度和测试数据长度：")
        print(len(train_data), len(test_data))

        # 添加到对应集合
        train_set = np.concatenate((train_set, train_data), axis=0)
        test_set = np.concatenate((test_set, test_data), axis=0)

    print("train_set.shape:")
    print(train_set.shape)
    print("test_set.shape:")
    print(test_set.shape)
    print(train_set[0])
    print(test_set[0])
    print("type(train_set)", type(train_set))


    x_train = train_set[:, 2:].astype(np.float32)
    x_test = test_set[:, 2:].astype(np.float32)
    y_train = train_set[:, 1:2].astype(np.float32)
    y_test = test_set[:, 1:2].astype(np.float32)

    print("x_train")
    print(x_train.shape)
    print("x_test")
    print(x_test.shape)
    print("y_train")
    print(y_train.shape)
    print("y_test")
    print(y_test.shape)
    print("-----------------")
    # 这里的X和Y对应是没问题的
    print(x_train[0])
    print(y_train[0])
    print(x_test[5], y_test[5])
    print("-----------------")

    return x_train, x_test, y_train, y_test



# 对特征值进行标准化/归一化操作
def preHandleDF(df_pre):
    #取出类别列和Y值列
    group_info = np.array(df_pre.iloc[:, :2])
    feature_info = np.array(df_pre.iloc[:, 2:])
    # 标准化、归一化
    feature_info = normalFunction(feature_info)
    # 水平拼接在一起
    concatenated_horizontal = np.concatenate((group_info, feature_info), axis=1)

    # 转成df
    df = pd.DataFrame(concatenated_horizontal)
    print("标准化/归一化后的df.shape：", df.shape)
    print("标准化/归一化后的df.info：", df.info)
    return df


def normalFunction(array):
    # X标准化
    scaler_standard = StandardScaler()
    result = scaler_standard.fit_transform(array)

    # X归一化
    x_scaler = MinMaxScaler(feature_range=(0, 1))
    result = x_scaler.fit_transform(result)
    return result






if __name__ == '__main__':
    # smiles = "CCCCCC"
    # preHandleData(smiles, 256)
    # descriptors = descriptersFromExcel()
    splitTrainTest("D:\\code\\PyCharm_WorkSpace\\ai4fuel\\data\\LHVDescriptors.xlsx")
    # getAllData("D:\\code\\PyCharm_WorkSpace\\ai4fuel\\data\\LHVDescriptors.xlsx")
