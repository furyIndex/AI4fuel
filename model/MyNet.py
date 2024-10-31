import torch.nn as nn
import torch.nn.functional as F




class MyNet(nn.Module):
    def __init__(self, out_dim=1, in_dim=256, dim1=128, dim2=64,
                 dropout_rate=0.3):
        super(MyNet, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout_rate = dropout_rate
        # 全连接层256->128
        self.fc1 = nn.Linear(in_dim, dim1)
        # 批量归一化
        self.bn1 = nn.BatchNorm1d(dim1)
        # dropout层减少过拟合
        self.dropout1 = nn.Dropout(dropout_rate)

        # # 新增的全连接层 dim1 -> dim2
        # self.fc2 = nn.Linear(dim1, dim2)
        # # 新增的dropout层
        # self.dropout2 = nn.Dropout(dropout_rate)

        # 全连接层，输出结果
        self.fc3 = nn.Linear(dim1, out_dim)

        # 权重初始化
        # self.fc_init()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)

        # # 新增的层操作
        # x = F.elu(self.fc2(x))
        # x = self.dropout2(x)

        x = self.fc3(x)
        return x

    # 初始化权重
    def fc_init(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)