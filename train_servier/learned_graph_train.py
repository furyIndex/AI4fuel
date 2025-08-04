import argparse
import ast
import copy
import os

import optuna
import pandas as pd
import torch
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader


from common.parse_args import args
from dataset import molDataset
from load_model.load_model_utils import showFig, showTruePred, predict_data
from model.transformerModel import SimpleTransformerRegressor
import numpy as np





def split_LG_Data(file_path, sheet_name):
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    data_length = df.shape[0]
    labels = df.iloc[:, 0].values
    labels = labels.reshape(-1, 1)
    data = df.iloc[:, 1:].values


    def parse_feature_values(data):
        return [np.array(ast.literal_eval(item)).astype('float32') for item in data]
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
    model.load_state_dict(state_dict)
    model.to(device)
    y_true, y_pred = predict_data(x_test, y_test, model)
    r2 = r2_score(y_test, y_pred)
    print("Saved model, calculate r2：", r2)

    # torch.save(model, '../save_models_server/{}_LG_{}_{}_model.pth'.format(file_name, sheet_name, r2))
    torch.save(model, '../save_models_server/{}_{}_{}_model.pth'.format(file_name, sheet_name, r2))

    path1 = "../result_png/{}_{}_R2.png".format(file_name, maxR2)
    path2 = "../result_png/{}_{}_TruePred.png".format(file_name, maxR2)


    showFig(file_name, y_test, y_pred, path1)
    showTruePred(file_name, y_test, y_pred, path2)





def objective(trial):
    global best_model_state
    global maxR2
    global X_test
    global Y_test
    global Y_pred
    global result_epoch


    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
    dim_feedforward = trial.suggest_categorical("dim_feedforward", [512, 1024])
    num_heads = trial.suggest_categorical('num_heads', [1, 2])
    num_layers = trial.suggest_int("num_layers", 1, 4)

    hidden_size_1 = trial.suggest_categorical("hidden_size_1", [64, 128])
    hidden_size_2 = trial.suggest_categorical("hidden_size_2", [1024])
    hidden_size_3 = trial.suggest_categorical("hidden_size_3", [128, 256])

    # epochs = trial.suggest_int("epochs", 150, 300, step=10)


    train_dataset = molDataset.MolDataset(x_train, y_train)


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


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
    # optimizer = torch.optim.Adam(transformModel.parameters(), lr=learning_rate, weight_decay=1e-3)


    loss_function = torch.nn.MSELoss()
    loss_function = loss_function.to(device)


    num_epochs = 400
    result_r2 = 0


    for epoch in range(num_epochs):

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



        transformModel.eval()
        with torch.no_grad():
            y_ture, y_pred = predict_data(x_test, y_test, transformModel)
            r2 = r2_score(y_test, y_pred)
            if r2 > maxR2:
                result_r2 = r2
                result_epoch = epoch
                maxR2 = r2
                best_model_state = copy.deepcopy(transformModel.state_dict())
                X_test = x_test
                Y_test = y_test
                Y_pred = y_pred

    return result_r2







if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument('--sheet_name', type=str, default='lg_k=10_a=0.1', help='fused data sheet name')
    parser.add_argument('--n_trials', type=int, default=100, help='trials of optuna study')
    parser.add_argument('--device', type=str, default="cuda:0", help='device')
    args = parser.parse_args()

    sheet_name = args.sheet_name
    trials = args.n_trials
    device = torch.device(args.device)


    data_dict = {
        '../data/HOV.xlsx' : 47,
        '../data/LHV.xlsx' : 47,
        '../data/RON.xlsx' : 49,
        '../data/MON.xlsx' : 49,
        '../data/CN.xlsx' : 49,
        '../data/YSI.xlsx' : 49,
        '../data/Density.xlsx' : 49,
        '../data/TB.xlsx' : 47,
        '../data/TM.xlsx' : 47,
        '../data/UFL.xlsx' : 47,
        '../data/LFL.xlsx' : 47,
        '../data/Viscosity.xlsx' : 47,
        '../data/Enthalpy_of_Vaporization.xlsx' : 47,
        '../data/VP.xlsx' : 47,
        '../data/DCN.xlsx' : 49,
        '../data/Surface_tension.xlsx' : 49,
        '../data/Flash_point.xlsx' : 47

    }

    optimal_results = []

    for file_path, seq_length in data_dict.items():


        file_name_with_extension = os.path.basename(file_path)
        file_name = os.path.splitext(file_name_with_extension)[0]
        study = optuna.create_study(direction='maximize', study_name=file_name)

        best_model_state = None
        maxR2 = 0
        result_epoch = 0
        X_test = []
        Y_test = []
        Y_pred = []


        x_train, x_test, y_train, y_test = split_LG_Data(file_path, sheet_name)
        all_data = np.vstack((x_train, x_test))
        study.optimize(objective, n_trials=trials)



        best_trial = study.best_trial


        save_model_fig(best_trial, best_model_state, seq_length, x_test, y_test)


        # with open('{}_{}_LG_{}.txt'.format(file_name, sheet_name, maxR2), 'w') as f:
        with open('{}_{}_{}.txt'.format(file_name, sheet_name, maxR2), 'w') as f:
            f.write(f"File: {file_path}\n")
            f.write(f"Best Value: {best_trial.value}\n")
            f.write(f"Best Params: {best_trial.params}\n")
            f.write(f"Best epoch: {result_epoch}\n")
            f.write('----------------\n')