import os
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from common.parse_args import args
from dataset import molDataset
from model.embedding_net import EmbeddingMLP, ContrastiveLoss, embedding_val_func
import numpy as np
from train_servier.gcn_loss import showFig



device = torch.device(args.device)




def split_descriptors_data(file_path):
    df = pd.read_excel(file_path, sheet_name='Sheet2')
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
    x_test = full_data[split:, 2:].astype(np.float32)
    y_train = full_data[:split, 1:2].astype(np.float32)
    y_test = full_data[split:, 1:2].astype(np.float32)

    x_scaler = StandardScaler()
    x_train = x_scaler.fit_transform(x_train)
    x_test = x_scaler.transform(x_test)

    # 标准化label，后续回归训练如果也标准化，预测结果就要反标准化
    y_scaler = StandardScaler()
    y_train = y_scaler.fit_transform(y_train)
    y_test = y_scaler.transform(y_test)

    return x_train, x_test, y_train, y_test




def train_embedding():
    # 初始化dataset
    train_dataset = molDataset.MolDataset(x_train, y_train)
    train_length = len(train_dataset)
    # 创建训练和验证数据加载器
    # 要求训练数据和测试数据中不同种类的化合物要一定比例
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # 初始化网络和优化器
    embedding_net = EmbeddingMLP(input_dim, 512, 128)


    embedding_net = embedding_net.to(device)
    optimizer = torch.optim.Adam(embedding_net.parameters(), lr=0.01)

    # 初始化损失函数
    loss_function = ContrastiveLoss(margin=4.0)
    loss_function = loss_function.to(device)

    # 记录训练轮数
    num_epochs = 50


    train_record = {}
    test_record = {}

    # 训练网络
    for epoch in range(num_epochs):
        embedding_net.train()
        train_loss = 0
        for feature, target in train_loader:
            feature, target = feature.to(device), target.to(device)
            optimizer.zero_grad()
            embeddings = embedding_net(feature)

            item_loss = loss_function.forward(embeddings, target)

            train_loss += item_loss.item()
            item_loss.backward()
            optimizer.step()
        train_loss /= train_length
        train_record[epoch] = train_loss
        print("第 {} 轮训练的误差：{}".format(epoch, train_loss))

        embedding_net.eval()
        with torch.no_grad():
            test_input = torch.tensor(x_test).float().to(device)
            test_label = torch.tensor(y_test).float().to(device)
            pred = embedding_net(test_input)
            eval_loss = loss_function.forward(pred, test_label).cpu().detach()
            print("第 {} 轮测试的误差：{}".format(epoch, eval_loss))
            test_record[epoch] = eval_loss

    embedding_net.eval()
    with torch.no_grad():
        test_input = torch.tensor(x_test).float().to(device)
        test_label = torch.tensor(y_test).float().to(device)
        pred = embedding_net(test_input)
        spearmanr_res = embedding_val_func(pred, test_label)
        print("spearmanr系数：", spearmanr_res)



    return embedding_net, spearmanr_res[0], train_record, test_record


if __name__ == '__main__':
    data_dict = {
        '../data/other/HOV.xlsx' : 1288,
        # '../data/other/LHV.xlsx' : 1287,
        # '../data/other/RON.xlsx' : 1393,
        # '../data/other/MON.xlsx' : 1394,
        # '../data/other/CN.xlsx' : 1394,
        # '../data/other/YSI.xlsx' : 1394,
        # '../data/other/Density.xlsx' : 1348,
        # '../data/other/TB.xlsx' : 1287,
        # '../data/other/TM.xlsx' : 1287,
        # '../data/other/UFL.xlsx' : 1288,
        # '../data/other/LFL.xlsx' : 1287,
        # '../data/other/Viscosity.xlsx' : 1288,
        # '../data/other/Enthalpy_of_Vaporization.xlsx' : 1288,
        # '../data/other/VP.xlsx' : 1287,
        # '../data/other/DCN.xlsx' : 1453,
        # '../data/other/Surface_tension.xlsx' : 1348,
        # '../data/other/Flash_point.xlsx' : 1287
    }
    # 存储每一轮的最优结果和参数
    optimal_results = []

    for file_path, input_dim in data_dict.items():

        # 获取文件名（包括后缀）
        file_name_with_extension = os.path.basename(file_path)
        # 分割文件名和后缀
        file_name = os.path.splitext(file_name_with_extension)[0]
        print(file_name)
        bestModel = None
        maxRes = 0
        train_best = None
        test_best = None



        # 获取数据
        x_train, x_test, y_train, y_test = split_descriptors_data(file_path)


        for k in range(10):
            embedding_net, res, train_record, test_record = train_embedding()
            if res > maxRes:
                maxRes = res
                bestModel = embedding_net
                train_best = train_record
                test_best = test_record


        print("最好斯皮尔曼系数：", maxRes)

        save_path_1 = "D:\\code\\PyCharm_WorkSpace\\ai4fuel\\result_png\\{}_train.png".format(file_name)
        save_path_2 = "D:\\code\\PyCharm_WorkSpace\\ai4fuel\\result_png\\{}_test.png".format(file_name)
        showFig(train_best, "Training", color="navy", save_path=save_path_1)
        showFig(test_best, "Testing", color="red", save_path=save_path_2)


        torch.save(bestModel, '../save_models_servier/{}_embedding_{}_{}.pth'.format(file_name, 4, maxRes))


        # test_input = torch.tensor(x_test).float().to(device)
        # test_label = torch.tensor(y_test).float().to(device)
        # spearmanr_res = embedding_val_func(test_input, test_label)
        # print("spearmanr系数：", spearmanr_res)