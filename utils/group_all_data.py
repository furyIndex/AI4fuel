import numpy as np
import pandas as pd


###############这个文件用来把去重smiles后的所有数据按照类别分类划分train/test，后续用来和预测后的结果拼接上#####################


def split_train_test(file_path, save_path):
    excel_file = pd.ExcelFile(file_path)
    sheet_names = excel_file.sheet_names  # 获取所有 sheet 名称

    for sheet_name in sheet_names:
        print("当前sheet：", sheet_name)
        df = pd.read_excel(file_path, sheet_name=sheet_name)

        # 按照类别分组
        grouped = df.groupby(df.columns[0])
        # 初始化空列表
        pre_data_len = len(df.columns)
        train_set = np.empty((0, pre_data_len))
        test_set = np.empty((0, pre_data_len))
        for name, group in grouped:
            # (group_len, 1290)
            group_data = group.values
            group_length = group_data.shape[0]
            # 打乱顺序
            np.random.seed(42)
            np.random.shuffle(group_data)
            # 划分训练集和测试集
            split = int(group_length * 0.8)
            train_data = group_data[:split, :]
            test_data = group_data[split:, :]
            # 添加到对应集合
            train_set = np.concatenate((train_set, train_data), axis=0)
            test_set = np.concatenate((test_set, test_data), axis=0)
        full_data = np.vstack((train_set, test_set))

        df = pd.DataFrame(full_data, columns=df.columns)
        with pd.ExcelWriter(save_path, engine='openpyxl', mode='a') as writer:
            # 如果文件不存在，mode='w' 会创建一个新文件，mode='a' 表示追加模式
            # 将 DataFrame 写入名为 'Sheet3' 的工作表，如果工作表已存在，将覆盖它
            df.to_excel(writer, sheet_name=sheet_name, index=False)





if __name__ == '__main__':
    file_path = "C:\\Users\\lx\\Desktop\\网站构建信息\\properties_data.xlsx"
    save_path = "C:\\Users\\lx\\Desktop\\网站构建信息\\grouped_properties_data.xlsx"
    split_train_test(file_path, save_path)


