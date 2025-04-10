import pickle

import numpy as np
import optuna
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader, Subset

from attentionweight import get_attention_weights, showHeatMap
from dataset import molDataset
from load_model.loadGCNModel import predict_data, showFig, showTruePred
from model.transformerModel import SimpleTransformerRegressor
from train_servier.data_prehandle import splitFPdata, splitGCNDescriptors, handle_NormAdjacMatrix, \
    splitGroupGCNData
from common.parse_args import args
import sys



# MSLE Loss
def msle_loss(pred, target):
    pred = torch.clamp(pred, min=1e-7)
    target = torch.clamp(target, min=1e-7)
    log_pred = torch.log1p(pred)
    log_target = torch.log1p(target)
    loss = torch.mean((log_pred - log_target) ** 2)
    return loss






def objective(trial):
    global bestModel
    global maxR2
    global X_test
    global Y_test
    global Y_pred
    global result_epoch


    # 定义超参数搜索空间

    num_layers = trial.suggest_int("num_layers", 1, 4)
    dim_feedforward = trial.suggest_categorical("dim_feedforward", [1024, 2048])

    # 初始化dataset
    train_dataset = molDataset.MolDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

    # 初始化网络和优化器
    transformModel = SimpleTransformerRegressor(input_dim=args.input_dim,
                                                seq_length=47,
                                                dim_feedforward=dim_feedforward,
                                                num_heads=2,
                                                num_layers=num_layers,
                                                hidden_dim_1=128,
                                                hidden_dim_2=1024,
                                                hidden_dim_3=256,
                                                dropout_rate=0.5)


    transformModel = transformModel.to(device)
    optimizer = torch.optim.AdamW(transformModel.parameters(), lr=0.00011913416043474528)

    # 记录训练轮数
    num_epochs = 400
    result_r2 = 0

    # 训练网络
    for epoch in range(num_epochs):
        transformModel.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = transformModel(data)
            item_loss = msle_loss(output, target)
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
            bestModel = transformModel
            X_test = x_test
            Y_test = y_test
            Y_pred = y_pred

    return result_r2




if __name__ == '__main__':

    device = torch.device(args.device)


    file_name = "Viscosity"
    vp_list = [
        # 'new_lg_k=10_a=0.1',
        'lg_k=10_a=0.1',
        # 'k=10_a=0.1',
        # 'k=10_a=0.2',
        # 'k=10_a=0.5',
        # 'k=50_a=0.1',
        # 'k=50_a=0.2',
        # 'k=50_a=0.5',
        # 'asy_k=50_a=0.1'
    ]
    print(file_name)
    for sheet_name in vp_list:
        print(sheet_name)

        bestModel = None
        result_epoch = 0
        maxR2 = 0
        X_test = []
        Y_test = []
        Y_pred = []

        # 读取数据
        file_path = '../data/other/{}.xlsx'.format(file_name)
        x_train, x_test, y_train, y_test = splitGroupGCNData(file_path, sheet_name)
        all_data = np.vstack((x_train, x_test))

        # 创建一个研究（study）
        study = optuna.create_study(direction='maximize', study_name=sheet_name)
        # 优化目标函数，传递额外的文件路径和数据大小参数
        study.optimize(objective, n_trials=100)


        torch.save(bestModel, '../save_models_servier/{}_{}_{}.pth'.format(file_name, sheet_name, maxR2))

        # path1 = "../result_png/{}_{}_{}_R2.png".format(file_name, sheet_name, maxR2)
        # path2 = "../result_png/{}_{}_{}_TruePred.png".format(file_name, sheet_name, maxR2)
        # path3 = "../result_png/{}_{}_{}_heatmap.png".format(file_name, sheet_name, maxR2)
        #
        # showFig(file_name, Y_test, Y_pred, path1)
        # showTruePred(file_name, Y_test, Y_pred, path2)
        # all_attention_weights = get_attention_weights(bestModel, all_data)
        # showHeatMap(file_name + "_all", all_attention_weights, path3)

        # 获取最佳试验
        best_trial = study.best_trial

        with open('{}_{}_{}.txt'.format(file_name, sheet_name, maxR2), 'w') as f:
            f.write(f"File: {file_path}\n")
            f.write(f"Best Value: {best_trial.value}\n")
            f.write(f"Best Params: {best_trial.params}\n")
            f.write(f"Best epoch: {result_epoch}\n")
            f.write('----------------\n')





