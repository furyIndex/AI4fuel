import math

import pandas as pd




def combain_column(column_name_list):
    all_data_file = "C:\\Users\\lx\\Desktop\\网站构建信息\\所有预测数据.xlsx"

    train_test_df = pd.read_excel(all_data_file, sheet_name="all_train_test_data")
    predict_df = pd.read_excel(all_data_file, sheet_name="all_predict_data")

    result_df = pd.DataFrame()
    for column_name in column_name_list:
        print(column_name)

        train_test_list = train_test_df[column_name].tolist()
        predict_list = predict_df[column_name].tolist()


        result_list = []
        for i in range(len(train_test_list)):
            if math.isnan(train_test_list[i]):
                result_list.append(predict_list[i])
            else:
                result_list.append(train_test_list[i])

        result_df[column_name] = result_list

    print(result_df.shape)
    print(result_df)

    with pd.ExcelWriter(all_data_file, engine='openpyxl', mode='a') as writer:
        # 如果文件不存在，mode='w' 会创建一个新文件，mode='a' 表示追加模式
        # 将 DataFrame 写入名为 'Sheet3' 的工作表，如果工作表已存在，将覆盖它
        result_df.to_excel(writer, sheet_name="merged_data", index=False)








if __name__ == '__main__':

    column_name_list = [
        'HOV_pre',
        'LHV_pre',
        'RON_pre',
        'MON_pre',
        'CN_pre',
        'YSI_pre',
        'Density_pre',
        'TB_pre',
        'TM_pre',
        'UFL_pre',
        'LFL_pre',
        'Viscosity_pre',
        'Enthalpy_of_Vaporization_pre',
        'VP_pre',
        'DCN_pre',
        'Surface_tension_pre',
        'Flash_point_pre',
    ]



    combain_column(column_name_list)