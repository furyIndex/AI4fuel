import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from dataset import molDataset
from model import MyNet
from torch.utils.tensorboard import SummaryWriter
from prepare import prepare_data
import numpy as np


'''
    用morgan指纹训练
'''

writer = SummaryWriter("logs")

# 准备数据
data = []
smiles_list = prepare_data.smilesFromExcel()
for smiles in smiles_list:
    data.append(prepare_data.getMorganFromSmiles(smiles, 256))
data = np.array(data)
labels = prepare_data.hoVFromExcel()

# 初始化dataset
dataset = molDataset.MolDataset(data, labels)

# 初始化KFold分割器
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 初始化网络和优化器
myNet = MyNet.MyNet(out_dim=1, in_dim=256, dim=128, dropout_rate=0.3)
myNet = myNet.cuda()
optimizer = torch.optim.Adam(myNet.parameters(), lr=0.001)

# 初始化损失函数
loss_function = torch.nn.MSELoss()
loss_function = loss_function.cuda()
# 初始化性能指标列表
performance_metrics = []

# 记录训练轮数
num_epochs = 1000



# 记录训练次数
total_train_step = 0
# 记录总的验证次数
total_eval_step = 0
# 记录交叉验证次数
kf_index = 1

# 4:1交叉验证
for train_index, val_index in kf.split(dataset):
    print("\n")
    print(f'====================第 {kf_index} 次交叉验证开始=====================')
    print("\n")
    # 创建训练和验证数据子集
    train_subset = Subset(dataset, train_index)
    val_subset = Subset(dataset, val_index)

    # 创建训练和验证数据加载器
    train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=64, shuffle=False)

    # 训练网络
    for epoch in range(num_epochs):
        print("--------------第{}次交叉验证，第 {} 轮训练开始--------------".format(kf_index, (epoch + 1)))
        myNet.train()
        for data, target in train_loader:
            data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = myNet(data)
            # 输出是[64,1,1]形状，转为一维的数据[64]
            output = output.view(-1)
            loss = loss_function(output.float(), target.float())
            loss.backward()
            optimizer.step()

            total_train_step += 1
            if total_train_step % 10 == 0:
                print("第{}次交叉验证，第 {} 轮，第 {} 次训练结束，训练误差为 {} ".format(kf_index, epoch + 1, total_train_step, loss.item()))
                writer.add_scalar('train_loss', loss.item(), total_train_step)

        # 验证网络
        myNet.eval()
        val_loss = 0
        with torch.no_grad():
            print("+++++++++++++++++第{}次交叉验证，第 {} 轮验证开始++++++++++++++++".format(kf_index, epoch + 1))
            for data, target in val_loader:
                data, target = data.cuda(), target.cuda()
                output = myNet(data)
                output = output.view(-1)
                val_loss += loss_function(output.float(), target.float()).item()

                total_eval_step += 1
                if total_eval_step % 10 == 0:
                    print("第{}次交叉验证，第 {} 轮，第{}次验证结束，此次验证误差为 {}".format(kf_index, epoch + 1, total_eval_step, val_loss))
                    writer.add_scalar('eval_loss', val_loss, total_eval_step)
        # 计算平均验证损失
        val_loss /= len(val_loader)
        print("第{}次交叉验证，第 {} 轮的平均验证误差 {}".format(kf_index, epoch+1, val_loss))

        # 记录性能指标
        performance_metrics.append(val_loss)

    kf_index += 1
# 计算平均性能指标
average_performance = sum(performance_metrics) / len(performance_metrics)
print(f'平均性能指标: {average_performance:.4f}')

print("======================================================")

val_loader = DataLoader(dataset, batch_size=64, shuffle=False)
val_loss = 0
with torch.no_grad():
    for data, target in val_loader:
        data, target = data.cuda(), target.cuda()
        output = myNet(data)
        output = output.view(-1)
        val_loss += loss_function(output.float(), target.float()).item()
val_loss /= len(val_loader)
print("最终预测误差：{}".format(val_loss))


torch.save(myNet, 'hoV_smiles_model.pth')

writer.close()