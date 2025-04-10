import pandas as pd





def save_all_fuel(file_path, save_path):
    excel_file = pd.ExcelFile(file_path)
    sheet_names = excel_file.sheet_names  # 获取所有 sheet 名称

    result_df = pd.DataFrame()

    for sheet_name in sheet_names:
        print("当前sheet：", sheet_name)
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        df_trimmed = df.iloc[:, :-2]
        result_df = pd.concat([df_trimmed, result_df], axis=0)


    print(result_df.shape[0])

    with pd.ExcelWriter(save_path, engine='openpyxl', mode='a') as writer:
        # 如果文件不存在，mode='w' 会创建一个新文件，mode='a' 表示追加模式
        # 将 DataFrame 写入名为 'Sheet3' 的工作表，如果工作表已存在，将覆盖它
        result_df.to_excel(writer, sheet_name="all_fuel", index=False)





def drop_duplicates_all_fuel(file_path):

    df = pd.read_excel(file_path, sheet_name="all_fuel")
    df_unique = df.drop_duplicates(subset=['SMILES'], keep='first')
    print(df_unique.shape[0])
    with pd.ExcelWriter(save_path, engine='openpyxl', mode='a') as writer:
        # 如果文件不存在，mode='w' 会创建一个新文件，mode='a' 表示追加模式
        # 将 DataFrame 写入名为 'Sheet3' 的工作表，如果工作表已存在，将覆盖它
        df_unique.to_excel(writer, sheet_name="all_fuel_unique", index=False)

















if __name__ == '__main__':
    file_path = "C:\\Users\\lx\\Desktop\\网站构建信息\\grouped_properties_data.xlsx"
    save_path = "C:\\Users\\lx\\Desktop\\网站构建信息\\all_fuel_data.xlsx"
    # save_all_fuel(file_path, save_path)
    drop_duplicates_all_fuel(save_path)