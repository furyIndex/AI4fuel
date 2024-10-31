import argparse




parser = argparse.ArgumentParser()



# =================args=====================#
parser.add_argument('--data_path', type=str,
                    default="D:\\code\\PyCharm_WorkSpace\\ai4fuel\\data\\CN_descriptors_values.xlsx",
                    help='input data path')
parser.add_argument('--preHandle_data_dim', type=int, default=1396, help='preHandle data dim')
parser.add_argument('--preHandle_data_normalization', type=int, default=0, help='preHandle data normalization')
parser.add_argument('--in_dim', type=int, default=1394, help='input data dim')
parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training')
parser.add_argument('--epochs', type=int, default=500, help='number of epochs to train')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--dim_1', type=int, default=128, help='dim_1')
parser.add_argument('--dim_2', type=int, default=32, help='dim_2')
parser.add_argument('--dropout_rate', type=float, default=0.5, help='dropout rate')
parser.add_argument('--weight_decay', type=float, default=0.001, help='weight decay')
parser.add_argument('--k_time', type=int, default=5, help='k_time')

# =================lhv args=====================#
# parser.add_argument('--data_path', type=str, default="D:\\code\\PyCharm_WorkSpace\\ai4fuel\\data\\LHVDescriptors.xlsx",
#                     help='input data path')
# parser.add_argument('--in_dim', type=int, default=1287, help='input data dim')
# parser.add_argument('--batch_size', type=int, default=128, help='input batch size for training')
# parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train')
# parser.add_argument('--lr', type=float, default=0.02, help='learning rate')
# parser.add_argument('--middle_layer', type=int, default=128, help='middle layer size')
# parser.add_argument('--dropout_rate', type=float, default=0.4, help='dropout rate')



args = parser.parse_args()