import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from dataset import molDataset
from model import MyNet
from torch.utils.tensorboard import SummaryWriter
from prepare import prepare_data
from prepare.min_max_scaler import TargetScaler
from common.parse_args import args
import sys


'''
    5折交叉验证
    1. 取5次训练中误差最小的那个模型（*）
    2. 在1的基础上用该模型将所有数据再训练一遍
    3. 保留5个模型，每次预测取这5个模型预测值的平均
'''

writer = SummaryWriter("logs")

# 准备数据
data = prepare_data.descriptersFromExcel(args.data_path)
labels = prepare_data.LHVFromExcel()
print("数据读取成功")
print(data.shape, labels.shape)

# Y值归一化
# scaler = TargetScaler(0, 1)
# labels_normalized = scaler.min_max_scaler.fit_transform(labels.reshape(-1, 1)).flatten()

# 初始化dataset
dataset = molDataset.MolDataset(data, labels)

# 初始化KFold分割器
kf = KFold(n_splits=5, shuffle=True, random_state=42)
# 记录交叉验证次数
kf_index = 1
# 初始化性能指标列表
performance_metrics = []
bestModel = None
minMSE = sys.maxsize
# 4:1交叉验证
for train_index, val_index in kf.split(dataset):

    # 初始化网络和优化器
    myNet = MyNet.MyNet(in_dim=args.in_dim, dim1=args.middle_layer, dropout_rate=args.dropout_rate)
    myNet = myNet.cuda()
    optimizer = torch.optim.Adam(myNet.parameters(), lr=args.lr)

    # 初始化损失函数
    loss_function = torch.nn.MSELoss()
    loss_function = loss_function.cuda()

    # 记录训练轮数
    num_epochs = args.epochs
    # 记录训练次数
    total_train_step = 0
    # 记录总的验证次数
    total_eval_step = 0

    print("\n")
    print(f'====================第 {kf_index} 次交叉验证开始=====================')
    print("\n")
    # 创建训练和验证数据子集
    train_subset = Subset(dataset, train_index)
    val_subset = Subset(dataset, val_index)

    # 创建训练和验证数据加载器
    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False)

    # 训练网络
    for epoch in range(num_epochs):
        print("--------------第{}次交叉验证，第 {} 轮训练开始--------------".format(kf_index, (epoch + 1)))
        myNet.train()
        for data, target in train_loader:
            data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            # print(data.shape)
            output = myNet(data)
            # 输出是[64,1,1]形状，转为一维的数据[64]
            output = output.view(-1)
            loss = loss_function(output.float(), target.float())
            loss.backward()
            optimizer.step()

            total_train_step += 1
            if total_train_step % 10 == 0:
                print("第{}次交叉验证，第 {} 轮，第 {} 次训练结束，训练误差为 {} ".format(kf_index, epoch + 1, total_train_step, loss.item()))
                writer.add_scalar('第{}次交叉验证train_loss'.format(kf_index), loss.item(), total_train_step)

        # 验证网络
        myNet.eval()
        val_loss = 0
        total_mse = 0
        total_samples = 0
        with torch.no_grad():
            print("+++++++++++++++++第{}次交叉验证，第 {} 轮验证开始++++++++++++++++".format(kf_index, epoch + 1))
            for data, target in val_loader:
                data, target = data.cuda(), target.cuda()
                output = myNet(data)
                output = output.view(-1)

                # 计算反归一化后的mse
                # batch_mse = scaler.batchMSE(output, target)
                # total_mse += batch_mse * len(target)
                # total_samples += len(target)

                # 计算反归一化前的mse
                val_loss += loss_function(output.float(), target.float()).item()
                total_eval_step += 1
                if total_eval_step % 10 == 0:
                    print("第{}次交叉验证，第 {} 轮，第{}次验证结束，此次验证误差为 {}".format(kf_index, epoch + 1, total_eval_step, val_loss))
                    writer.add_scalar('第{}次交叉验证eval_loss'.format(kf_index), val_loss, total_eval_step)
        # 计算平均验证损失
        val_loss /= len(val_loader)
        if val_loss < minMSE:
            minMSE = val_loss
            bestModel = myNet
        # print("第{}次交叉验证，第 {} 轮的平均验证误差 {}".format(kf_index, epoch+1, total_mse / total_samples))
        print("第{}次交叉验证，第 {} 轮的平均验证误差 {}".format(kf_index, epoch+1, val_loss))
        writer.add_scalar('第{}次交叉验证total_eval_loss'.format(kf_index), val_loss, epoch)
        # 记录性能指标
        performance_metrics.append(val_loss)

    kf_index += 1
# 计算平均性能指标
average_performance = sum(performance_metrics) / len(performance_metrics)
print(f'平均性能指标: {average_performance:.4f}')




torch.save(bestModel, 'hoV_descriptors_model_{}.pth'.format(minMSE))

writer.close()