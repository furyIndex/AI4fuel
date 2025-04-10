import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score

from common.parse_args import args
from load_model.loadGCNModel import predict_data
from train_servier.data_prehandle import splitGroupGCNData



###############这个文件用来把lg_k=10_a=0.1的预测结果全部保存下来和原值做对比#####################



device = torch.device(args.device)



def load_model(saved_model_path):
    """
    加载保存的模型
    :param saved_model_path: 模型保存路径
    :return: 加载的模型
    """
    model = torch.load(saved_model_path, map_location=device)
    model.eval()  # 设置为评估模式
    return model



def find_files_with_string_in_name(folder_path, search_string):
    path = ""
    # 遍历文件夹中的所有文件
    for file in os.listdir(folder_path):
        if search_string in file:  # 检查文件名是否包含特定字符串
            file_path = os.path.join(folder_path, file)  # 获取文件的完整路径
            path = file_path
    return path





def batch_load(save_path):

    # 设置data文件夹的路径
    data_folder_path = 'D:\\code\\PyCharm_WorkSpace\\ai4fuel\\data\\other'
    model_folder_path = "E:\\ai4fuel\\超参数训练\\gcn-transformer\\true-simple-mlp\\学习图训练\\可复现k=10_a=0.1\\模型"

    # 检查路径是否存在
    if os.path.exists(data_folder_path) and os.path.isdir(data_folder_path):
        # 遍历data文件夹中的文件
        for filename in os.listdir(data_folder_path):
            data_name = os.path.splitext(filename)[0]
            print(data_name)
            data_path = os.path.join(data_folder_path, filename)
            model_path = find_files_with_string_in_name(model_folder_path, data_name)
            print(model_path)


            myNet = load_model(model_path)
            myNet = myNet.to(device)

            x_train, x_test, y_train, y_test = splitGroupGCNData(data_path, "lg_k=10_a=0.1")
            all_data = np.vstack((x_train, x_test))
            all_labels = np.vstack((y_train, y_test))

            y_true, y_pred = predict_data(x_test, y_test, myNet)
            y_true = y_true.reshape(-1, 1)
            y_pred = y_pred.reshape(-1, 1)

            r2 = r2_score(y_true, y_pred)
            print(data_name + " 测试数据R2： ", r2)

            y_all_true, y_all_pred = predict_data(all_data, all_labels, myNet)
            y_all_true = y_all_true.flatten()
            y_all_pred = y_all_pred.flatten()

            df = pd.DataFrame({
                'y_all_true': y_all_true,
                'y_all_pred': y_all_pred
            })

            with pd.ExcelWriter(save_path, engine='openpyxl', mode='a') as writer:
                # 如果文件不存在，mode='w' 会创建一个新文件，mode='a' 表示追加模式
                # 将 DataFrame 写入名为 'Sheet3' 的工作表，如果工作表已存在，将覆盖它
                df.to_excel(writer, sheet_name=data_name, index=False)




def single_load(save_path):

    filename = "CN"
    model_path = "E:\\ai4fuel\\超参数训练\\gcn-transformer\\true-simple-mlp\\学习图训练\\可复现k=10_a=0.1\\模型\\CN_LG_lg_k=10_a=0.1_0.8978158235549927_model.pth"
    data_path = "/data/CN.xlsx"


    myNet = load_model(model_path)
    myNet = myNet.to(device)

    x_train, x_test, y_train, y_test = splitGroupGCNData(data_path, "lg_k=10_a=0.1")
    all_data = np.vstack((x_train, x_test))
    all_labels = np.vstack((y_train, y_test))
    print("数据读取成功")

    y_true, y_pred = predict_data(x_test, y_test, myNet)

    r2 = r2_score(y_true, y_pred)
    print(filename + " 测试数据R2： ", r2)

    y_all_true, y_all_pred = predict_data(all_data, all_labels, myNet)
    y_all_true = y_all_true.flatten()
    y_all_pred = y_all_pred.flatten()

    df = pd.DataFrame({
        'y_all_true': y_all_true,
        'y_all_pred': y_all_pred
    })

    with pd.ExcelWriter(save_path, engine='openpyxl', mode='a') as writer:
        # 如果文件不存在，mode='w' 会创建一个新文件，mode='a' 表示追加模式
        # 将 DataFrame 写入名为 'Sheet3' 的工作表，如果工作表已存在，将覆盖它
        df.to_excel(writer, sheet_name=filename, index=False)



if __name__ == '__main__':
    save_path = "D:\\code\\PyCharm_WorkSpace\\ai4fuel\\data\\predict_data.xlsx"
    # single_load(save_path)
    batch_load(save_path)

