import os
import pickle
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from common.parse_args import args
from descriptors_group.getGCNDescriptors import knn_train, l1_norm, norm_adj_train
from load_model.loadGCNModel import load_model


device = torch.device(args.device)





# 划分transformer的描述符数据
def save_x_train(file_name, file_path):
    df = pd.read_excel(file_path, sheet_name='Sheet2')
    all_columns = df.columns
    descriptor_name = all_columns[2:].tolist()
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
    full_data = np.vstack((train_set, test_set))
    split = int(full_data.shape[0] * 0.8)
    x_train = full_data[:split, 2:].astype(np.float32)

    # 保存到文件
    # with open("../result/{}_train_data.pkl".format(file_name), "wb") as f:
    #     pickle.dump({"x_train": x_train, "descriptor_name": descriptor_name}, f)

    return x_train




########################保存embedding后的gcn特征#########################
def save_train_A(file_name, file_path, model, k, a):

    x_train = save_x_train(file_name, file_path)

    # 标准化特征
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)


    # embedding
    train_input = torch.tensor(x_train).to(device)
    train_embedding = model(train_input)
    train_embedding = train_embedding.cpu().detach().numpy()

    with open("../result/{}_train_embedding_data.pkl".format(file_name), "wb") as f:
        pickle.dump({"train_embedding": train_embedding}, f)


    # # 获取相似性矩阵
    # train_similarity_matrix = np.dot(train_embedding, train_embedding.T)
    #
    # # 训练数据的特征融合
    # train_knn_matrix = knn_train(train_similarity_matrix, k)
    # train_norm_matrix = l1_norm(train_knn_matrix)
    # train_A, train_degree = norm_adj_train(train_norm_matrix, a)
    #
    # # 加载后还需要乘上原数据
    # # train_gcn_matrix = np.dot(train_A, x_train)
    #
    # # 保存到文件
    # with open("../result/{}_train_A_data.pkl".format(file_name), "wb") as f:
    #     pickle.dump({"train_A": train_A, "train_degree": train_degree}, f)







if __name__ == '__main__':


    file_list = {
        '../data/other/HOV.xlsx',
        '../data/other/LHV.xlsx',
        '../data/other/RON.xlsx',
        '../data/other/MON.xlsx',
        '../data/other/CN.xlsx',
        '../data/other/YSI.xlsx',
        '../data/other/Density.xlsx',
        '../data/other/TB.xlsx',
        '../data/other/TM.xlsx',
        '../data/other/UFL.xlsx',
        '../data/other/LFL.xlsx',
        '../data/other/Viscosity.xlsx',
        '../data/other/Enthalpy_of_Vaporization.xlsx',
        '../data/other/VP.xlsx',
        '../data/other/DCN.xlsx',
        '../data/other/Surface_tension.xlsx',
        '../data/other/Flash_point.xlsx',

    }

    embedding_dict = {
        '../data/other/HOV.xlsx' : "HOV_embedding_2_0.5871895242769624.pth",
        '../data/other/LHV.xlsx' : "LHV_embedding_3_0.8772815444910687.pth",
        '../data/other/RON.xlsx' : "RON_embedding_1_0.3760063468414003.pth",
        '../data/other/MON.xlsx' : "MON_embedding_1_0.3775167319125256.pth",
        '../data/other/CN.xlsx' : "CN_embedding_1_0.3480172106524897.pth",
        '../data/other/YSI.xlsx' : "YSI_embedding_1_0.8152132600867458.pth",
        '../data/other/Density.xlsx' : "Density_embedding_1_0.48608576201740794.pth",
        '../data/other/TB.xlsx' : "TB_embedding_1_0.45683157386267037.pth",
        '../data/other/TM.xlsx' : "TM_embedding_3_0.526635736508536.pth",
        '../data/other/UFL.xlsx' : "UFL_embedding_3_0.6188434893620851.pth",
        '../data/other/LFL.xlsx' : "LFL_embedding_3_0.7880042580593524.pth",
        '../data/other/Viscosity.xlsx' : "Viscosity_embedding_1_0.7457670924015943.pth",
        '../data/other/Enthalpy_of_Vaporization.xlsx' : "Enthalpy_of_Vaporization_embedding_3_0.8097393628062347.pth",
        '../data/other/VP.xlsx' : "VP_embedding_1_0.833519575858319.pth",
        '../data/other/DCN.xlsx' : "DCN_embedding_3_0.5978619025504474.pth",
        '../data/other/Surface_tension.xlsx' : "Surface_tension_embedding_1_0.5136778129492438.pth",
        '../data/other/Flash_point.xlsx' : "Flash_point_embedding_1_0.4566019803325665.pth"

    }

    print(file_list)

    embedding_folder = "C:\\Users\\lx\\Desktop\\图表\\模型文件\\最好效果-学习图-k=10_a=0.1\\embedding模型\\"

    for file_path in file_list:
        print(file_path)
        embedding_path = embedding_folder + embedding_dict[file_path]
        print(embedding_path)
        embedding_net = load_model(embedding_path)
        embedding_net = embedding_net.to(device)

        # 获取文件名（包括后缀）
        file_name_with_extension = os.path.basename(file_path)
        # 分割文件名和后缀
        file_name = os.path.splitext(file_name_with_extension)[0]
        print(file_name)

        save_train_A(file_name, file_path, embedding_net, k=10, a=0.1)






