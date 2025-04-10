import pandas as pd

if __name__ == '__main__':
    prediction_data_file = "C:\\Users\\lx\\Desktop\\网站构建信息\\每类属性预测值数据库.xlsx"
    all_data_file = "C:\\Users\\lx\\Desktop\\网站构建信息\\所有预测数据.xlsx"
    all_df = pd.read_excel(all_data_file, sheet_name="all_fuel_unique")

    sheet_name_list = [
        'HOV',
        'LHV',
        'RON',
        'MON',
        'CN',
        'YSI',
        'Density',
        'Tb',
        'Tm',
        'UFL',
        'LFL',
        'Viscosity',
        'Enthalpy_of_Vaporization',
        'VP',
        'DCN',
        'Surface_tension',
        'Flash_point'
    ]

    for sheet_name in sheet_name_list:
        print(sheet_name)
        prediction_df = pd.read_excel(prediction_data_file, sheet_name=sheet_name)

        # 创建SMILES到文件A第5列数据的字典映射（注意列索引从0开始）
        smiles_pred_dict = dict(zip(prediction_df['SMILES'], prediction_df['y_all_pred']))

        all_df["{}_pre".format(sheet_name)] = all_df['SMILES'].map(smiles_pred_dict)

    with pd.ExcelWriter(all_data_file, engine='openpyxl', mode='a') as writer:
        # 如果文件不存在，mode='w' 会创建一个新文件，mode='a' 表示追加模式
        # 将 DataFrame 写入名为 'Sheet3' 的工作表，如果工作表已存在，将覆盖它
        all_df.to_excel(writer, sheet_name="all_data", index=False)

