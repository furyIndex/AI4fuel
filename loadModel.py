import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from dataset import molDataset, testDataset
from model.MyNet import MyNet
from torch.utils.tensorboard import SummaryWriter
from prepare import prepare_data
from prepare.prepare_data import splitTrainTest, getAllData
from prepare.min_max_scaler import TargetScaler
from common.parse_args import args
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from prepare.prepare_data import normalFunction


myNet = torch.load(
    "D:\\code\\PyCharm_WorkSpace\\ai4fuel\\sava_models\\CN_descriptors_model_2.6108845930833082.pth")
myNet = myNet.cuda()
x, y = getAllData(args.data_path)
print("数据读取成功")
len = x.shape[0]

# 标准化、归一化
x = normalFunction(x)

dataset = molDataset.MolDataset(x, y)
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

# train_dataset = molDataset.MolDataset(x_train, y_train)
# test_dataset = molDataset.MolDataset(x_test, y_test)
# train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

loss_function = torch.nn.MSELoss()
loss_function = loss_function.cuda()

myNet.eval()
test_loss = 0
with torch.no_grad():
    for data, target in dataloader:
        data, target = data.cuda(), target.cuda()
        output = myNet(data)
        # print("output", output)
        # print("target", target)
        # 计算反归一化前的mse
        item_loss = loss_function(output.float(), target.float()).item()
        test_loss += item_loss

# 计算平均验证损失
test_loss /= len
print("平均验证误差 {}".format(test_loss))
