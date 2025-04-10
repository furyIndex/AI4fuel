import pickle

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader, Subset
from dataset import molDataset
from model.transformerModel import TransformerRegressor, SimpleTransformerRegressor
from torch.utils.tensorboard import SummaryWriter
from prepare.prepare_data import splitTrainTest, getAllData, normalFunction, handle_PCA
from train_servier.data_prehandle import splitFPdata, splitGCNDescriptors, getAdjacMatrix, handle_NormAdjacMatrix, \
    splitGroupGCNData
from common.parse_args import args
import sys




def showFig(data, name, color):
    # 获取epoch和loss数据
    epochs = list(data.keys())
    losses = list(data.values())

    # 创建折线图
    plt.figure(figsize=(10, 6))  # 设置图像大小
    plt.plot(epochs, losses, color=color, label='{} Loss'.format(name))  # 绘制训练过程折线图

    # 添加标题和标签
    plt.title('{} Loss Over Epochs'.format(name))
    plt.xlabel('Epoch', fontsize=16)
    plt.ylabel('Loss', fontsize=16)

    # 添加图例
    plt.legend()

    # 设置y轴为对数刻度
    plt.yscale('log')

    # 显示网格
    plt.grid(True)

    # 显示图像
    plt.show()
    plt.close()




def showTwoFig(train_data, test_data):

    # 获取epoch和loss数据
    train_epochs = list(train_data.keys())
    train_losses = list(train_data.values())
    test_epochs = list(test_data.keys())
    test_losses = list(test_data.values())

    # 创建折线图
    plt.figure(figsize=(10, 6))  # 设置图像大小
    plt.plot(train_epochs, train_losses, marker='o', label='Training Loss')  # 绘制训练过程折线图
    plt.plot(test_epochs, test_losses, marker='s', label='Testing Loss')  # 绘制测试过程折线图

    # 添加标题和标签
    plt.title('Training and Testing Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    # 添加图例
    plt.legend()

    # 显示网格
    plt.grid(True)

    # 显示图像
    plt.show()



# MSLE Loss
def msle_loss(pred, target):
    pred = torch.clamp(pred, min=1e-7)
    target = torch.clamp(target, min=1e-7)
    log_pred = torch.log1p(pred)
    log_target = torch.log1p(target)
    loss = torch.mean((log_pred - log_target) ** 2)
    return loss



if __name__ == '__main__':

    device = torch.device(args.device)

    vp_list = [
        'k=10_a=0.1',
        # 'k=10_a=0.2',
        # 'k=10_a=0.5',
        # 'k=50_a=0.1',
        # 'k=50_a=0.2',
        # 'k=50_a=0.5',
        # 'asy_k=50_a=0.1'
    ]

    for sheet_name in vp_list:
        print(sheet_name)

        bestModel = None
        maxR2 = 0
        best_k = 0

        # 读取数据
        file_path = '../data/other/VP.xlsx'
        x_train, x_test, y_train, y_test = splitGroupGCNData(file_path, sheet_name)
        print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)


        # y_train = np.log(y_train + 1)


        for k in range(1):

            train_record = {}
            test_record = {}

            # 初始化dataset
            train_dataset = molDataset.MolDataset(x_train, y_train)
            test_dataset = molDataset.MolDataset(x_test, y_test)


            train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

            train_length = len(train_dataset)

            # 初始化网络和优化器
            transformModel = SimpleTransformerRegressor(input_dim=args.input_dim,
                                                        seq_length=47,
                                                        dim_feedforward=1338,
                                                        num_heads=2,
                                                        num_layers=1,
                                                        hidden_dim_1=128,
                                                        hidden_dim_2=2048,
                                                        hidden_dim_3=512,
                                                        dropout_rate=0.5)


            transformModel = transformModel.to(device)
            optimizer = torch.optim.AdamW(transformModel.parameters(), lr=0.00024845724103621236, weight_decay=1e-4)

            # 初始化损失函数
            loss_function = torch.nn.MSELoss()
            loss_function = loss_function.to(device)


            # 记录训练轮数
            num_epochs = 500
            # 记录训练次数
            total_train_step = 0
            # 记录总的验证次数
            total_eval_step = 0


            # 训练网络
            for epoch in range(num_epochs):
                # print("--------------第{}次，第 {} 轮训练开始--------------".format(k, (epoch + 1)))
                transformModel.train()
                train_loss = 0
                for data, target in train_loader:
                    data, target = data.to(device), target.to(device)
                    optimizer.zero_grad()
                    output = transformModel(data)

                    item_loss = msle_loss(output, target)

                    train_loss += item_loss.item()
                    item_loss.backward()
                    optimizer.step()
                    total_train_step += 1
                train_loss /= train_length
                # 记录训练数据
                train_record[epoch] = train_loss
                # print("第{}次，第 {} 轮的平均训练误差 {}".format(k, epoch+1, train_loss))

                # 验证网络
                transformModel.eval()
                with torch.no_grad():
                    # print("+++++++++++++++++第{}次，第 {} 轮验证开始++++++++++++++++".format(k, epoch + 1))
                    y_true, y_pred = predict_data(x_test, y_test, transformModel)
                    mse = mean_squared_error(y_true, y_pred)
                    r2 = r2_score(y_test, y_pred)
                test_record[epoch] = mse
                if (epoch+1) % 50 == 0:
                    print("第{}次，第 {} 轮的平均测试R2 {}".format(k, epoch+1, r2))

            transformModel.eval()
            with torch.no_grad():
                # 计算整体误差
                y_ture, y_pred = predict_data(x_test, y_test, transformModel)
                r2 = r2_score(y_test, y_pred)
                if r2 > maxR2:
                    maxR2 = r2
                    bestModel = transformModel
                    best_k = k

            showFig(train_record, "Training", color="navy")
            showFig(test_record, "Testing", color="red")

            k = k + 1
        print("最后r2：", maxR2)
        print("最好k：", best_k)
        print("===============================================")
        # torch.save(bestModel, '../save_models_servier/VP_{}_{}.pth'.format (sheet_name, maxR2))




