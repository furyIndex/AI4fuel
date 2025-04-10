import os

import numpy as np
import torch
from sklearn.metrics import r2_score, mean_absolute_error

from attentionweight import get_attention_weights, showHeatMap
from common.parse_args import args
from load_model.loadGCNModel import load_model, predict_data, showFig, showTruePred
from train_servier.learned_graph_train import split_LG_Data



device = torch.device(args.device)

def find_files_starting_with_string(folder_path, search_string):
    path = ""
    # 遍历文件夹中的所有文件
    for file in os.listdir(folder_path):
        if file.startswith(search_string):  # 检查文件名是否以特定字符串开头
            file_path = os.path.join(folder_path, file)  # 获取文件的完整路径
            path = file_path  # 更新路径（最终返回最后一个匹配项）
    return path








if __name__ == '__main__':

    # data_folder_path = "../data/other"
    data_folder_path = "../data/temp"
    model_folder_path = "C:\\Users\\lx\\Desktop\\图表\\模型文件\\最好效果-学习图-k=10_a=0.1\\gcn-transformer\\模型"


    if os.path.exists(data_folder_path) and os.path.isdir(data_folder_path):
        # 遍历data文件夹中的文件
        for filename in os.listdir(data_folder_path):
            data_name = os.path.splitext(filename)[0]
            print(data_name)
            data_path = os.path.join(data_folder_path, filename)
            model_path = find_files_starting_with_string(model_folder_path, data_name)
            print(model_path)


            x_train, x_test, y_train, y_test = split_LG_Data(data_path, "lg_k=10_a=0.1")
            all_data = np.vstack((x_train, x_test))

            myNet = load_model(model_path)
            myNet = myNet.to(device)

            y_true, y_pred = predict_data(x_test, y_test, myNet)
            r2 = r2_score(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            print("R2：", r2)
            print("MAE: ", mae)


            path1 = "../result_png/r2pic/{}_R2.png".format(data_name)
            path2 = "../result_png/r2pic/{}_TruePred.png".format(data_name)
            path4 = "../result_png/r2pic/{}_all_heatmap.png".format(data_name)


            showFig(data_name, y_test, y_pred, path1)
            showTruePred(data_name, y_test, y_pred, path2)
            # 所有数据的注意力得分
            all_attention_weights = get_attention_weights(myNet, all_data)
            showHeatMap(data_name, all_attention_weights, path4)