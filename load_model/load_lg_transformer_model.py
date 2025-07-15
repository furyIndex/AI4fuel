import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score

from common.parse_args import args
from load_model.loadGCNModel import predict_data, split_LG_Data

device = torch.device(args.device)



def load_model(saved_model_path):
    model = torch.load(saved_model_path, map_location=device)
    model.eval()
    return model



def find_files_with_string_in_name(folder_path, search_string):
    path = ""
    for file in os.listdir(folder_path):
        if search_string in file:
            file_path = os.path.join(folder_path, file)
            path = file_path
    return path





def batch_load(save_path):


    data_folder_path = '../data/other'
    model_folder_path = "../save_models_server"


    if os.path.exists(data_folder_path) and os.path.isdir(data_folder_path):

        for filename in os.listdir(data_folder_path):
            data_name = os.path.splitext(filename)[0]
            print(data_name)
            data_path = os.path.join(data_folder_path, filename)
            model_path = find_files_with_string_in_name(model_folder_path, data_name)
            print(model_path)


            myNet = load_model(model_path)
            myNet = myNet.to(device)

            x_train, x_test, y_train, y_test = split_LG_Data(data_path, "lg_k=10_a=0.1")
            all_data = np.vstack((x_train, x_test))
            all_labels = np.vstack((y_train, y_test))

            y_true, y_pred = predict_data(x_test, y_test, myNet)
            y_true = y_true.reshape(-1, 1)
            y_pred = y_pred.reshape(-1, 1)

            r2 = r2_score(y_true, y_pred)
            print(data_name + " predict R2： ", r2)

            y_all_true, y_all_pred = predict_data(all_data, all_labels, myNet)
            y_all_true = y_all_true.flatten()
            y_all_pred = y_all_pred.flatten()

            df = pd.DataFrame({
                'y_all_true': y_all_true,
                'y_all_pred': y_all_pred
            })

            with pd.ExcelWriter(save_path, engine='openpyxl', mode='a') as writer:
                df.to_excel(writer, sheet_name=data_name, index=False)




def single_load(save_path):

    filename = "CN"
    model_path = "../save_models_server"
    data_path = "../data/CN.xlsx"


    myNet = load_model(model_path)
    myNet = myNet.to(device)

    x_train, x_test, y_train, y_test = split_LG_Data(data_path, "lg_k=10_a=0.1")
    all_data = np.vstack((x_train, x_test))
    all_labels = np.vstack((y_train, y_test))
    print("Data read successfully")

    y_true, y_pred = predict_data(x_test, y_test, myNet)

    r2 = r2_score(y_true, y_pred)
    print(filename + " predict R2： ", r2)

    y_all_true, y_all_pred = predict_data(all_data, all_labels, myNet)
    y_all_true = y_all_true.flatten()
    y_all_pred = y_all_pred.flatten()

    df = pd.DataFrame({
        'y_all_true': y_all_true,
        'y_all_pred': y_all_pred
    })

    with pd.ExcelWriter(save_path, engine='openpyxl', mode='a') as writer:
        df.to_excel(writer, sheet_name=filename, index=False)



if __name__ == '__main__':
    save_path = "../data/predict_data.xlsx"
    batch_load(save_path)

