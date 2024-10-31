import numpy as np
from sklearn.linear_model import LinearRegression
from prepare import prepare_data
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
from prepare.prepare_data import splitTrainTest
from common.parse_args import args
from sklearn.model_selection import train_test_split

'''
    调用sklearn包对比
'''


# 准备数据

# 根据class划分
x_train, x_test, y_train, y_test = splitTrainTest(args.data_path)


# 随机划分
# features_df = pd.read_excel('D:\\code\\PyCharm_WorkSpace\\ai4fuel\\data\\LHVDescriptors.xlsx', sheet_name='Sheet1')
# labels_df = pd.read_excel('D:\\code\\PyCharm_WorkSpace\\ai4fuel\\data\\LHV_unique.xlsx', sheet_name='Sheet1')
# features_df.reset_index(drop=True, inplace=True)
# labels_df.reset_index(drop=True, inplace=True)
# labels_data = labels_df['LHV']
# if isinstance(labels_data, pd.Series):
#     labels_data = labels_data.to_frame()
# x_train, x_test, y_train, y_test = train_test_split(features_df, labels_data, test_size=0.2, random_state=42)


print("数据读取成功")


# Y值归一化
# min_max_scaler = MinMaxScaler(feature_range=(0, 1))
# y_train = min_max_scaler.fit_transform(y_train)



# 创建回归模型
model = LinearRegression()
model.fit(x_train, y_train)

# 进行预测
y_pred = model.predict(x_test)

# 反归一化
# y_pred = min_max_scaler.inverse_transform(y_pred)

print("y_pred", y_pred)
print("y_test", y_test)
# MSE误差
mse = mean_squared_error(y_test, y_pred)
print("MSE误差:", mse)


