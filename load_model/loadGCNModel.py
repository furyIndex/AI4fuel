import ast
import os

import pandas as pd
import torch


from attentionweight import showHeatMap, get_attention_weights

from common.parse_args import args

import matplotlib.pyplot as plt

from sklearn.metrics import r2_score

import numpy as np

from train_servier.data_prehandle import getAdjacMatrix, handle_NormAdjacMatrix, splitGroupGCNData

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

    # labels标准化，因为训练embedding net的时候对label标准化了，后续预测的时候需要反标准化
    # label_scaler = StandardScaler()
    # y_train = label_scaler.fit_transform(y_train)
    # y_test = label_scaler.transform(y_test)


    return x_train, x_test, y_train, y_test






def predict_data(x, y, myNet):
    x = torch.tensor(x)
    y = torch.tensor(y)
    x = x.to(device)

    myNet.eval()
    output = myNet(x)
    output = output.cpu().detach().numpy()
    y = y.cpu()
    y = y.detach().numpy()
    y = np.array(y)
    predict_data = np.array(output)
    return y, predict_data





def showFig(name, y_true, y_pred, save_path):

    # 计算 R² 值
    r2 = r2_score(y_true, y_pred)
    print("R2：", r2)
    # 确保 y_true 和 y_pred 是一维数组
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    # 设置图形大小
    plt.figure(figsize=(10, 6))

    # 绘制散点图
    plt.scatter(y_true, y_pred, color='skyblue', edgecolor='black', alpha=0.7, label='Data Points')

    # 绘制对角线（理想预测线）
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=2, label='Ideal Prediction')

    # 绘制回归线
    coef = np.polyfit(y_true, y_pred, 1)  # 线性回归系数
    poly1d_fn = np.poly1d(coef)  # 构造回归函数
    plt.plot(y_true, poly1d_fn(y_true), color='green', linestyle='-', linewidth=2, label='Regression Line')

    # 设置标题和标签
    plt.title(f'{name}\nR² = {r2:.4f}', fontsize=22, fontweight='bold', pad=20)
    plt.xlabel('Actual', fontsize=18, fontweight='bold')
    plt.ylabel('Predicted', fontsize=18, fontweight='bold')
    plt.tick_params(axis='both', which='major', labelsize=16)  # 主刻度

    # 设置网格线
    plt.grid(True, linestyle='--', alpha=0.7)

    # 设置图例
    plt.legend(loc='upper left', fontsize=16)

    # 调整布局
    plt.tight_layout()

    # 保存高质量图像
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

    # 显示图形
    plt.show()
    plt.close()



def showTruePred(name, y_true, y_pred, save_path):
    # 设置图形大小
    plt.figure(figsize=(10, 6))

    # 绘制真实值和预测值
    plt.plot(y_true, label='True Value', color='blue', marker='o', markersize=8, linestyle='-', linewidth=2, alpha=0.8)
    plt.plot(y_pred, label='Predicted Value', color='red', marker='s', markersize=8, linestyle='--', linewidth=2, alpha=0.8)

    # 设置标题和标签
    plt.title(f'{name}: True Value and Predicted Value', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Sample Index', fontsize=14, fontweight='bold')
    plt.ylabel('Value', fontsize=14, fontweight='bold')

    # 设置网格线
    plt.grid(True, linestyle='--', alpha=0.7)

    # 设置图例
    plt.legend(loc='upper right', fontsize=12)

    # 调整布局
    plt.tight_layout()

    # 保存高质量图像
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

    # 显示图形
    plt.show()
    plt.close()




def load_model(saved_model_path):
    """
    加载保存的模型
    :param saved_model_path: 模型保存路径
    :return: 加载的模型
    """
    model = torch.load(saved_model_path, map_location=device)
    model.eval()  # 设置为评估模式
    return model


def mean_squared_error(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2)
    return mse

def mae_func(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def single_test():


    model_name = "Surface_tension_LG_lg_k=10_a=0.1_0.939228892326355_model.pth"
    data_name = "Surface_tension"

    # model_path = "./save_models_servier/" + model_name
    model_path = "C:\\Users\\lx\\Desktop\\图表\\模型文件\\最好效果-学习图-k=10_a=0.1\\gcn-transformer\\模型\\" + model_name
    # model_path = "C:\\Users\\lx\\Desktop\\图表\\模型文件\\最好效果-手动图-gcn_k=10_a=0.1\\" + model_name

    # file_path = 'D:\\code\\PyCharm_WorkSpace\\ai4fuel\\data\\other\\{}.xlsx'.format(data_name)
    file_path = './data/other/{}.xlsx'.format(data_name)

    print(model_path)
    print(file_path)

    myNet = load_model(model_path)
    myNet = myNet.to(device)

    x_train, x_test, y_train, y_test = split_LG_Data(file_path, "lg_k=10_a=0.1")

    all_data = np.vstack((x_train, x_test))
    print("数据读取成功")

    # 输入转为tensor
    y_true, y_pred = predict_data(x_test, y_test, myNet)


    r2 = r2_score(y_true, y_pred)
    mae = mae_func(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    print("总mae:", mae)
    print("总mse:", mse)
    print("总r2是：", r2)

    x_test = torch.tensor(x_test)


    path1 = "./result_png/{}_R2.png".format(data_name)
    path2 = "./result_png/{}_TruePred.png".format(data_name)
    # path3 = "./result_png/{}_test_heatmap.png".format(data_name)
    path4 = "./result_png/{}_all_heatmap.png".format(data_name)


    showFig(data_name, y_test, y_pred, path1)
    showTruePred(data_name, y_test, y_pred, path2)
    # test的注意力得分
    # test_attention_weights = get_attention_weights(myNet, x_test)
    # showHeatMap(data_name + "_test", test_attention_weights, path3)
    # 所有数据的注意力得分
    all_attention_weights = get_attention_weights(myNet, all_data)
    showHeatMap(data_name + "_all", all_attention_weights, path4)


def batch_test():

    # 设置data文件夹的路径
    data_folder_path = '/data/other'
    model_folder_path = "C:\\Users\\lx\\Desktop\\图表\\模型文件\\k=10_a=0.1模型"

    # 检查路径是否存在
    if os.path.exists(data_folder_path) and os.path.isdir(data_folder_path):
        # 遍历data文件夹中的文件
        for filename in os.listdir(data_folder_path):
            data_name = os.path.splitext(filename)[0]
            print(data_name)
            data_path = os.path.join(data_folder_path, filename)
            model_path = find_files_with_string_in_name(model_folder_path, data_name)
            print(data_path)
            print(model_path)


            myNet = load_model(model_path)
            myNet = myNet.to(device)
            # print(myNet)
            x_train, x_test, y_train, y_test = splitGroupGCNData(data_path, "k=10_a=0.1")
            all_data = np.vstack((x_train, x_test))
            print("数据读取成功")

            path1 = "./result_png/{}_R2.png".format(data_name)
            path2 = "./result_png/{}_TruePred.png".format(data_name)
            # path3 = "./result_png/{}_test_heatmap.png".format(data_name)
            path4 = "./result_png/{}_all_heatmap.png".format(data_name)

            y_ture, y_pred = predict_data(x_test, y_test, myNet)

            showFig(data_name, y_test, y_pred, path1)
            showTruePred(data_name, y_test, y_pred, path2)
            # test的注意力得分
            # test_attention_weights = get_attention_weights(myNet, x_test)
            # showHeatMap(data_name + "_test", test_attention_weights, path3)
            # 所有数据的注意力得分
            all_attention_weights = get_attention_weights(myNet, all_data)
            showHeatMap(data_name + "_all", all_attention_weights, path4)

            


def find_files_with_string_in_name(folder_path, search_string):
    path = ""
    # 遍历文件夹中的所有文件
    for file in os.listdir(folder_path):
        if search_string in file:  # 检查文件名是否包含特定字符串
            file_path = os.path.join(folder_path, file)  # 获取文件的完整路径
            path = file_path
    return path





if __name__ == '__main__':
    '''
    获取all文件夹下所有文件对应模型的r2、热力图等
    '''
    # device = torch.device("cpu")

    single_test()
    # batch_test()
