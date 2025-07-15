import ast
import pandas as pd
import torch
from common.parse_args import args
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import numpy as np

device = torch.device(args.device)




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


    r2 = r2_score(y_true, y_pred)
    print("R2：", r2)

    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()


    plt.figure(figsize=(10, 6))

    plt.scatter(y_true, y_pred, color='skyblue', edgecolor='black', alpha=0.7, label='Data Points')

    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=2, label='Ideal Prediction')

    coef = np.polyfit(y_true, y_pred, 1)
    poly1d_fn = np.poly1d(coef)
    plt.plot(y_true, poly1d_fn(y_true), color='green', linestyle='-', linewidth=2, label='Regression Line')

    plt.title(f'{name}\nR² = {r2:.4f}', fontsize=22, fontweight='bold', pad=20)
    plt.xlabel('Actual', fontsize=18, fontweight='bold')
    plt.ylabel('Predicted', fontsize=18, fontweight='bold')
    plt.tick_params(axis='both', which='major', labelsize=16)

    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper left', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    # plt.show()
    plt.close()





def showFig2(name, y_true, y_pred, save_path):


    r2 = r2_score(y_true, y_pred)
    print("R2：", r2)

    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, color='skyblue', edgecolor='black', alpha=0.7, label='Data Points')
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=2, label='Ideal Prediction')
    coef = np.polyfit(y_true, y_pred, 1)
    poly1d_fn = np.poly1d(coef)
    plt.plot(y_true, poly1d_fn(y_true), color='green', linestyle='-', linewidth=2, label='Regression Line')
    plt.title(f'{name}\nR² = {r2:.4f}', fontsize=24, fontweight='bold', pad=20)
    plt.xlabel('Actual', fontsize=22, fontweight='bold')
    plt.ylabel('Predicted', fontsize=22, fontweight='bold')
    plt.tick_params(axis='both', which='major', labelsize=20)

    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper left', fontsize=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()
    plt.close()








def showTruePred(name, y_true, y_pred, save_path):

    plt.figure(figsize=(10, 6))


    plt.plot(y_true, label='True Value', color='blue', marker='o', markersize=8, linestyle='-', linewidth=2, alpha=0.8)
    plt.plot(y_pred, label='Predicted Value', color='red', marker='s', markersize=8, linestyle='--', linewidth=2, alpha=0.8)

    plt.title(f'{name}: True Value and Predicted Value', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Sample Index', fontsize=14, fontweight='bold')
    plt.ylabel('Value', fontsize=14, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper right', fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    # plt.show()
    plt.close()




def load_model(saved_model_path):
    model = torch.load(saved_model_path, map_location=device)
    model.eval()
    return model


def mean_squared_error(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2)
    return mse

def mae_func(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

