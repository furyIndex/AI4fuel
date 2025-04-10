import os
from optuna.visualization import plot_parallel_coordinate
import optuna
import torch
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader

from attentionweight import compute_attention_weights, showHeatMap, get_attention_weights
from common.parse_args import args
from dataset import molDataset
from load_model.loadGCNModel import showFig, showTruePred, predict_data
from model.MLPNet import MLPNet
from data_prehandle import splitTrainTest, splitFPdata, splitGCNDescriptors, getAdjacMatrix, handle_NormAdjacMatrix, \
    splitGroupGCNData
from model.transformerModel import TransformerRegressor, SimpleTransformerRegressor
import numpy as np

device = torch.device(args.device)

def objective(trial, x_train, x_test, y_train, y_test):
    global bestModel
    global maxR2
    global X_test
    global Y_test
    global Y_pred
    # 定义超参数搜索空间
    params = {
        # "batch_size": trial.suggest_int("batch_size", 32, 128, step=16),
        "num_heads": trial.suggest_categorical("num_heads", [1, 3, 6]),
        "num_layers": trial.suggest_categorical("num_layers", [1, 2]),
        "dim_feedforward": trial.suggest_categorical("dim_feedforward", [1024]),
        "hidden_dim_1": trial.suggest_categorical("hidden_dim_1", [64, 128]),
        "hidden_dim_2": trial.suggest_int("hidden_dim_2", 256, 1024, step=128),
        "hidden_dim_3": trial.suggest_categorical("hidden_dim_3", [32, 64, 128]),
        # "dropout_rate": trial.suggest_float("dropout_rate", 0.5, 0.5),
        "learning_rate": trial.suggest_float("learning_rate", 5e-4, 5e-2, log=True),
    }
    # epochs = trial.suggest_int("epochs", 500, 800, step=50)


    # 初始化dataset
    train_dataset = molDataset.MolDataset(x_train, y_train)

    # 创建训练和验证数据加载器
    # 要求训练数据和测试数据中不同种类的化合物要一定比例
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    # 初始化网络和优化器
    transformModel = SimpleTransformerRegressor(input_dim=args.input_dim,
                                                seq_length=seq_length,
                                                dim_feedforward=params["dim_feedforward"],
                                                num_heads=params["num_heads"],
                                                num_layers=params["num_layers"],
                                                hidden_dim_1=params["hidden_dim_1"],
                                                hidden_dim_2=params["hidden_dim_2"],
                                                hidden_dim_3=params["hidden_dim_3"],
                                                dropout_rate=0.5)
    transformModel = transformModel.to(device)
    optimizer = torch.optim.Adam(transformModel.parameters(), lr=params["learning_rate"])

    # 初始化损失函数
    loss_function = torch.nn.MSELoss()
    loss_function = loss_function.to(device)

    # 记录训练轮数
    num_epochs = 200


    # 训练网络
    for epoch in range(num_epochs):
        transformModel.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = transformModel(data)
            item_loss = loss_function(output.float(), target.float())
            item_loss.backward()
            optimizer.step()
    transformModel.eval()
    with torch.no_grad():
        # 测试网络
        y_ture, y_pred = predict_data(x_test, y_test, transformModel)
        r2 = r2_score(y_test, y_pred)

    if trial.number == 0:
        bestModel = transformModel
        maxR2 = r2
        X_test = x_test
        Y_test = y_test
        Y_pred = y_pred
    else:
        if r2 > maxR2:
            bestModel = transformModel
            maxR2 = r2
            X_test = x_test
            Y_test = y_test
            Y_pred = y_pred
    return r2






if __name__ == '__main__':
    data_dict = {
        # '../data/other/HOV.xlsx' : 47,
        # '../data/other/LHV.xlsx' : 47,
        # '../data/other/RON.xlsx' : 49,
        # '../data/other/MON.xlsx' : 49,
        # '../data/other/CN.xlsx' : 49,
        # '../data/other/YSI.xlsx' : 49,
        # '../data/other/Density.xlsx' : 49,
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

    for file_path, seq_length in data_dict.items():

        # 获取文件名（包括后缀）
        file_name_with_extension = os.path.basename(file_path)
        # 分割文件名和后缀
        file_name = os.path.splitext(file_name_with_extension)[0]
        # 创建一个研究（study）
        study = optuna.create_study(direction='maximize', study_name=file_name)

        bestModel = None
        maxR2 = 0
        X_test = []
        Y_test = []
        Y_pred = []

        # 获取数据
        x_train, x_test, y_train, y_test = splitGroupGCNData(file_path, 'k=50_a=1')
        all_data = np.vstack((x_train, x_test))
        # 优化目标函数，传递额外的文件路径和数据大小参数
        study.optimize(lambda t: objective(t, x_train, x_test, y_train, y_test), n_trials=50)

        fig = plot_parallel_coordinate(study)
        fig.show()

        # torch.save(bestModel, '../save_models_servier/{}_k=50_a=1_{}.pth'.format(file_name, maxR2))
        # path1 = "../result_png/{}_k=50_a=1_{}_R2.png".format(file_name, maxR2)
        # path2 = "../result_png/{}_k=50_a=1_{}_TruePred.png".format(file_name, maxR2)
        # path3 = "../result_png/{}_k=50_a=1_{}_heatmap.png".format(file_name, maxR2)
        #
        # showFig(file_name, Y_test, Y_pred, path1)
        # showTruePred(file_name, Y_test, Y_pred, path2)
        # all_attention_weights = get_attention_weights(bestModel, all_data)
        # showHeatMap(file_name + "_all", all_attention_weights, path3)
        #
        # # 获取最佳试验
        # best_trial = study.best_trial
        #
        # with open('{}_optimal_results.txt'.format(file_name), 'w') as f:
        #     f.write(f"File: {file_path}\n")
        #     f.write(f"Best Value: {best_trial.value}\n")
        #     f.write(f"Best Params: {best_trial.params}\n")
        #     f.write('----------------\n')
