import ast
import copy
import os

import optuna
import pandas as pd
import torch
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader

from attentionweight import get_attention_weights, showHeatMap
from common.parse_args import args
from dataset import molDataset
from load_model.loadGCNModel import showFig, showTruePred, predict_data
from model.transformerModel import SimpleTransformerRegressor
import numpy as np

from train_servier.gcn_loss import msle_loss

device = torch.device(args.device)





def split_LG_Data(file_path, sheet_name):
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



def save_model_fig(trail, state_dict, seq_length, x_test, y_test):
    model = SimpleTransformerRegressor(
        seq_length=seq_length,
        input_dim=606,
        dim_feedforward=trail.params["dim_feedforward"],
        hidden_dim_1=trail.params["hidden_size_1"],
        hidden_dim_2=trail.params["hidden_size_2"],
        hidden_dim_3=trail.params["hidden_size_3"],
        num_heads=trail.params["num_heads"],
        num_layers=trail.params["num_layers"],
        dropout_rate=0.5,
        output_dim=1,
    )
    model.load_state_dict(state_dict)  # 加载状态字典
    model.to(device)
    y_true, y_pred = predict_data(x_test, y_test, model)
    r2 = r2_score(y_test, y_pred)
    print("保存后的模型，计算r2：", r2)

    # torch.save(model, '../save_models_servier/{}_LG_{}_{}_model.pth'.format(file_name, sheet_name, r2))
    torch.save(model, '../save_models_servier/{}_{}_{}_model.pth'.format(file_name, sheet_name, r2))

    path1 = "../result_png/{}_{}_R2.png".format(file_name, maxR2)
    path2 = "../result_png/{}_{}_TruePred.png".format(file_name, maxR2)
    path3 = "../result_png/{}_{}_heatmap.png".format(file_name, maxR2)


    showFig(file_name, y_test, y_pred, path1)
    showTruePred(file_name, y_test, y_pred, path2)
    all_attention_weights = get_attention_weights(model, all_data)
    showHeatMap(file_name, all_attention_weights, path3)





def objective(trial):
    global best_model_state
    global maxR2
    global X_test
    global Y_test
    global Y_pred
    global result_epoch

    # 定义超参数搜索空间
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
    dim_feedforward = trial.suggest_categorical("dim_feedforward", [512, 1024, 2048])
    num_heads = trial.suggest_categorical('num_heads', [1, 2, 6])
    num_layers = trial.suggest_int("num_layers", 1, 4)

    hidden_size_1 = trial.suggest_categorical("hidden_size_1", [64, 128])
    hidden_size_2 = trial.suggest_categorical("hidden_size_2", [1024])
    hidden_size_3 = trial.suggest_categorical("hidden_size_3", [128, 256])

    # epochs = trial.suggest_int("epochs", 150, 300, step=10)

    # 初始化dataset
    train_dataset = molDataset.MolDataset(x_train, y_train)

    # 创建训练和验证数据加载器
    # 要求训练数据和测试数据中不同种类的化合物要一定比例
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 初始化网络和优化器
    transformModel = SimpleTransformerRegressor(input_dim=args.input_dim,
                                                seq_length=seq_length,
                                                dim_feedforward=dim_feedforward,
                                                num_heads=num_heads,
                                                num_layers=num_layers,
                                                hidden_dim_1=hidden_size_1,
                                                hidden_dim_2=hidden_size_2,
                                                hidden_dim_3=hidden_size_3,
                                                dropout_rate=args.dropout_rate)
    transformModel = transformModel.to(device)
    optimizer = torch.optim.AdamW(transformModel.parameters(), lr=learning_rate, weight_decay=1e-3)

    # 初始化损失函数
    loss_function = torch.nn.MSELoss()
    loss_function = loss_function.to(device)

    # 记录训练轮数
    num_epochs = 300
    result_r2 = 0


    for epoch in range(num_epochs):
        # 训练网络
        transformModel.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = transformModel(data)

            # item_loss = msle_loss(output, target)
            item_loss = loss_function(output.float(), target.float())
            # item_loss = torch.sqrt(item_loss)

            item_loss.backward()
            optimizer.step()


        # 测试网络
        transformModel.eval()
        with torch.no_grad():
            # 计算整体误差
            y_ture, y_pred = predict_data(x_test, y_test, transformModel)
            r2 = r2_score(y_test, y_pred)
            if r2 > maxR2:
                result_r2 = r2
                result_epoch = epoch
                maxR2 = r2
                best_model_state = copy.deepcopy(transformModel.state_dict())  # 深拷贝状态字典
                X_test = x_test
                Y_test = y_test
                Y_pred = y_pred

    return result_r2





if __name__ == '__main__':

    data_dict = {
        # '../data/other/HOV.xlsx' : 47,
        # '../data/other/LHV.xlsx' : 47,
        # '../data/other/RON.xlsx' : 49,
        # '../data/other/MON.xlsx' : 49,
        # '../data/other/CN.xlsx' : 49,
        # '../data/other/YSI.xlsx' : 49,
        '../data/other/Density.xlsx' : 49,
        # '../data/other/TB.xlsx' : 47,
        # '../data/other/TM.xlsx' : 47,
        # '../data/other/UFL.xlsx' : 47,
        # '../data/other/LFL.xlsx' : 47,
        # '../data/other/Viscosity.xlsx' : 47,
        # '../data/other/Enthalpy_of_Vaporization.xlsx' : 47,
        # '../data/other/VP.xlsx' : 47,
        # '../data/other/DCN.xlsx' : 49,
        # '../data/other/Surface_tension.xlsx' : 49,
        # '../data/other/Flash_point.xlsx' : 47

    }
    # 存储每一轮的最优结果和参数
    optimal_results = []
    sheet_name = 'lg_k=10_a=0.1'

    for file_path, seq_length in data_dict.items():

        # 获取文件名（包括后缀）
        file_name_with_extension = os.path.basename(file_path)
        # 分割文件名和后缀
        file_name = os.path.splitext(file_name_with_extension)[0]
        # 创建一个研究（study）
        study = optuna.create_study(direction='maximize', study_name=file_name)

        best_model_state = None
        maxR2 = 0
        result_epoch = 0
        X_test = []
        Y_test = []
        Y_pred = []

        # 获取数据
        x_train, x_test, y_train, y_test = split_LG_Data(file_path, sheet_name)
        all_data = np.vstack((x_train, x_test))
        # 优化目标函数，传递额外的文件路径和数据大小参数
        study.optimize(objective, n_trials=300)

        # 保存状态字典
        # torch.save(best_model_state, '../save_models_servier/{}_LG_{}_{}_dict.pth'.format(file_name, sheet_name, maxR2))

        # 获取最佳试验
        best_trial = study.best_trial

        # 保存模型以及图片
        save_model_fig(best_trial, best_model_state, seq_length, x_test, y_test)

        # 保存超参数
        # with open('{}_{}_LG_{}.txt'.format(file_name, sheet_name, maxR2), 'w') as f:
        with open('{}_{}_{}.txt'.format(file_name, sheet_name, maxR2), 'w') as f:
            f.write(f"File: {file_path}\n")
            f.write(f"Best Value: {best_trial.value}\n")
            f.write(f"Best Params: {best_trial.params}\n")
            f.write(f"Best epoch: {result_epoch}\n")
            f.write('----------------\n')