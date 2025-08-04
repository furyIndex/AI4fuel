import os

import numpy as np
import torch
from sklearn.metrics import r2_score, mean_absolute_error
from common.parse_args import args
from load_model.load_model_utils import load_model, predict_data, showFig, showTruePred, showFig2
from train_servier.learned_graph_train import split_LG_Data



device = torch.device(args.device)

def find_files_starting_with_string(folder_path, search_string):
    path = ""
    for file in os.listdir(folder_path):
        if file.startswith(search_string):
            file_path = os.path.join(folder_path, file)
            path = file_path
    return path








if __name__ == '__main__':



    data_folder_path = "../data"
    model_folder_path = "../save_models_server/lg_transformer"


    if os.path.exists(data_folder_path) and os.path.isdir(data_folder_path):

        for filename in os.listdir(data_folder_path):
            print("=============================================")
            data_name = os.path.splitext(filename)[0]
            print(data_name)
            data_path = os.path.join(data_folder_path, filename)
            model_path = find_files_starting_with_string(model_folder_path, data_name)


            x_train, x_test, y_train, y_test = split_LG_Data(data_path, "lg_k=10_a=0.1")
            all_data = np.vstack((x_train, x_test))

            myNet = load_model(model_path)
            myNet = myNet.to(device)

            y_true_test, y_pred_test = predict_data(x_test, y_test, myNet)
            y_true_train, y_pred_train = predict_data(x_train, y_train, myNet)

            r2_train = r2_score(y_true_train, y_pred_train)
            print("train R2: ", r2_train)

            r2_test = r2_score(y_true_test, y_pred_test)
            print("test R2: ", r2_test)

