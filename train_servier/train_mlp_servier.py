import os
import sys

import pandas as pd
import optuna
import torch
from torch.utils.data import DataLoader
from common.parse_args import args
from dataset import molDataset
from loadModel import predict_data, mean_squared_error
from model.MLPNet import MLPNet
from data_prehandle import splitTrainTest
import joblib


def objective(trial, data_path, data_dim):
    # 定义超参数搜索空间
    learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-3, log=True)
    batch_size = trial.suggest_int("batch_size", 32, 256)
    hidden_size_1 = trial.suggest_int("hidden_size_1", 500, 1000)
    hidden_size_2 = trial.suggest_int("hidden_size_2", 64, 500)
    epochs = trial.suggest_int("epochs", 100, 1000)


    # 准备数据
    x_train, x_test, y_train, y_test = splitTrainTest(data_path, data_dim)

    # 初始化dataset
    train_dataset = molDataset.MolDataset(x_train, y_train)

    # 创建训练和验证数据加载器
    # 要求训练数据和测试数据中不同种类的化合物要一定比例
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 初始化网络和优化器
    myNet = MLPNet(in_dim=data_dim, dim1=hidden_size_1, dim2=hidden_size_2, dropout_rate=args.dropout_rate)
    myNet = myNet.cuda()
    optimizer = torch.optim.Adam(myNet.parameters(), lr=learning_rate, weight_decay=args.weight_decay)

    # 初始化损失函数
    loss_function = torch.nn.MSELoss()
    loss_function = loss_function.cuda()

    # 记录训练轮数
    num_epochs = epochs

    # 训练网络
    for epoch in range(num_epochs):
        myNet.train()
        for data, target in train_loader:
            data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = myNet(data)
            item_loss = loss_function(output.float(), target.float())
            item_loss.backward()
            optimizer.step()

    myNet.eval()
    with torch.no_grad():
        # 测试网络
        y_ture, y_pred = predict_data(x_test, y_test, myNet)
        mse = mean_squared_error(y_ture, y_pred)
    return mse


if __name__ == '__main__':
    data_dict = {
        # '../data/hovDescriptors.xlsx': 1288,
        '../data/LHVDescriptors.xlsx': 1287,
        '../data/RON.xlsx': 1393,
        '../data/MON.xlsx': 1394,
        '../data/CN.xlsx': 1394,
    }
    # 存储每一轮的最优结果和参数
    optimal_results = []

    for file_path, data_size in data_dict.items():
        print(file_path)
        print(data_size)

        # 获取文件名（包括后缀）
        file_name_with_extension = os.path.basename(file_path)
        # 分割文件名和后缀
        file_name = os.path.splitext(file_name_with_extension)[0]

        # 创建一个研究（study）
        study = optuna.create_study(study_name=file_name, direction='minimize')
        # 优化目标函数，传递额外的文件路径和数据大小参数
        study.optimize(lambda t: objective(t, file_path, data_size), n_trials=700)
        # 获取最佳试验
        best_trial = study.best_trial
        # 将最优结果和参数存储到字典中
        optimal_results.append({
            'file_path': file_path,
            'data_size': data_size,
            'best_value': best_trial.value,
            'best_params': best_trial.params
        })
    # 打印所有最优结果
    for result in optimal_results:
        print(f"File: {result['file_path']}")
        print(f"Data Size: {result['data_size']}")
        print(f"Best Value: {result['best_value']}")
        print(f"Best Params: {result['best_params']}")
        print('---------------------------')
    # 保存到文本文件
    with open('optimal_results.txt', 'w') as f:
        for result in optimal_results:
            f.write(f"File: {result['file_path']}\n")
            f.write(f"Data Size: {result['data_size']}\n")
            f.write(f"Best Value: {result['best_value']}\n")
            f.write(f"Best Params: {result['best_params']}\n")
            f.write('----------------\n')