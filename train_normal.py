import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from dataset import molDataset, testDataset
from model.MyNet import MyNet
from torch.utils.tensorboard import SummaryWriter
from prepare import prepare_data
from prepare.prepare_data import splitTrainTest
from prepare.min_max_scaler import TargetScaler
from common.parse_args import args
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
import sys

'''
    按类别训练
'''

writer = SummaryWriter("logs")

bestModel = None
minMSE = sys.maxsize
loss_list = []

for k in range(args.k_time):

    # 准备数据
    x_train, x_test, y_train, y_test = splitTrainTest(args.data_path)
    print("数据读取成功")
    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
    train_length = x_train.shape[0]
    test_length = x_test.shape[0]

    # Y值归一化，后面测试得到的y要反归一化
    # min_max_scaler = MinMaxScaler(feature_range=(0, 1))
    # y_train = min_max_scaler.fit_transform(y_train)

    # 初始化dataset
    train_dataset = molDataset.MolDataset(x_train, y_train)
    test_dataset = molDataset.MolDataset(x_test, y_test)

    # 创建训练和验证数据加载器
    # 要求训练数据和测试数据中不同种类的化合物要一定比例
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # 初始化网络和优化器
    myNet = MyNet(in_dim=args.in_dim, dim1=args.dim_1, dim2=args.dim_2, dropout_rate=args.dropout_rate)
    myNet = myNet.cuda()
    optimizer = torch.optim.Adam(myNet.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # 初始化损失函数
    loss_function = torch.nn.MSELoss()
    loss_function = loss_function.cuda()
    # 初始化性能指标列表
    performance_metrics = []

    # 记录训练轮数
    num_epochs = args.epochs
    # 记录训练次数
    total_train_step = 0
    # 记录总的验证次数
    total_eval_step = 0

    temp = 0
    # 训练网络
    for epoch in range(num_epochs):
        print("--------------第{}次，第 {} 轮训练开始--------------".format(k, (epoch + 1)))
        myNet.train()
        train_loss = 0
        for data, target in train_loader:
            data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = myNet(data)
            item_loss = loss_function(output.float(), target.float())
            train_loss += item_loss.item()
            item_loss.backward()
            optimizer.step()
            total_train_step += 1
            # if total_train_step % 10 == 0:
            #     print("第{}次，第 {} 轮，第 {} 次训练结束，训练误差为 {} ".format(k, epoch + 1, total_train_step, item_loss.item()/args.batch_size))
            #     writer.add_scalar('{}_train_loss'.format(k), item_loss.item()/args.batch_size, total_train_step)
        train_loss /= train_length
        print("第{}次，第 {} 轮的平均测试误差 {}".format(k, epoch+1, train_loss))
        writer.add_scalar('{}_train_loss'.format(k), train_loss, epoch + 1)

        # 验证网络
        myNet.eval()
        test_loss = 0
        with torch.no_grad():
            print("+++++++++++++++++第{}次，第 {} 轮验证开始++++++++++++++++".format(k, epoch + 1))
            for data, target in test_loader:
                data, target = data.cuda(), target.cuda()
                output = myNet(data)
                # 计算反归一化前的mse
                item_loss = loss_function(output.float(), target.float()).item()
                test_loss += item_loss
                total_eval_step += 1
        # 计算平均验证损失
        test_loss /= test_length
        temp = test_loss
        print("第{}次，第 {} 轮的平均验证误差 {}".format(k, epoch+1, test_loss))
        writer.add_scalar('{}_eval_loss'.format(k), test_loss, epoch + 1)
        # 记录性能指标
        performance_metrics.append(test_loss)
    if temp < minMSE:
        minMSE = temp
        bestModel = myNet
    loss_list.append(temp)
    # 计算平均性能指标
    average_performance = sum(performance_metrics) / len(performance_metrics)
    print(f'平均性能指标: {average_performance:.4f}')
    k = k + 1


# print("=========================反归一化前验证=============================")
# myNet.eval()
# total_mse = 0
# with torch.no_grad():
#     for data, target in test_loader:
#         # data, target = data.cuda(), target.cuda()
#         output = myNet(data)
#         # 反归一化
#
#         item_loss = loss_function(output, target).item()
#         total_mse += item_loss
# total_mse /= test_length
# print("反归一化的mse", total_mse)


j = 1
for i in loss_list:
    print("第{}次MSE：".format(j), i)
    j += 1
torch.save(myNet, 'D:\\code\\PyCharm_WorkSpace\\ai4fuel\\sava_models\\CN_descriptors_model_{}.pth'.format(minMSE))

writer.close()