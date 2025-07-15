import argparse

parser = argparse.ArgumentParser()



parser.add_argument('--device', type=str, default="cuda:0", help='device')



# =================transformer args=====================#
parser.add_argument('--data_path', type=str,
                    default="D:\\code\\PyCharm_WorkSpace\\ai4fuel\\data\\HOV.xlsx",
                    help='input data path')
parser.add_argument('--data_dim', type=int, default=49, help='data dim')
parser.add_argument('--input_dim', type=int, default=606, help='nn input dim')
parser.add_argument('--preHandle_data_normalization', type=int, default=1, help='preHandle data normalization')
parser.add_argument('--preHandle_data_PCA', type=int, default=0, help='PCA trigger')
parser.add_argument('--dim_feedforward', type=int, default=1024, help='feedforward dimension')
parser.add_argument('--dim_1', type=int, default=64, help='dim_1')
parser.add_argument('--dim_2', type=int, default=16, help='dim_2')
parser.add_argument('--num_heads', type=int, default=1, help='head number')
parser.add_argument('--num_layers', type=int, default=2, help='layer number')

parser.add_argument('--batch_size', type=int, default=184, help='input batch size for training')
parser.add_argument('--epochs', type=int, default=500, help='number of epochs to train')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
parser.add_argument('--dropout_rate', type=float, default=0.5, help='dropout rate')
parser.add_argument('--weight_decay', type=float, default=0.001, help='weight decay')
parser.add_argument('--k_time', type=int, default=1, help='k_time')



# =================gcn args=====================#


parser.add_argument('--alpha', type=float, default=0.1, help='alpha')
parser.add_argument('--k', type=int, default=10, help='knn')













args = parser.parse_args()