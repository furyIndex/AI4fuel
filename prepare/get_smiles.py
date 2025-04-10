import pubchempy as pcp
import pandas as pd


'''
获取每一个化合物的smiles表达式
'''
def getSmilesFromCAS(file_path):
    # 读取Excel文件
    df = pd.read_excel(file_path, sheet_name='Sheet1')
    smiles_list = []
    # 访问列
    cas_data = df['CAS'].values.tolist()
    cas_list = []
    # 去除读取的excel中不规范的cas值（比如有的读出来是0）
    for cas in cas_data:
        if type(cas) is str:
            cas_list.append(cas)
        else:
            print(cas_data.index(cas))
    print("有效CAS长度： ", len(cas_list))
    i = 2
    for cas in cas_list:
        print(i)
        print(cas)
        compounds = pcp.get_compounds(cas, 'name')
        print(len(compounds))
        if len(compounds) == 1:
            compound = compounds[0]
            print(compound.isomeric_smiles)
            # 如果单个化合物的smiles为空，就填空
            if compound.isomeric_smiles is None:
                smiles_list.append("null")
                print("null")
            else:
                smiles_list.append(compound.isomeric_smiles)
        if len(compounds) == 0:
            smiles_list.append("null")
            print("null")
        # 如果查出来的化合物仍然有多种，就把他们的smiles拼一起
        if len(compounds) > 1:
            smiles_list.append("multiple")
        print("列表长度：{}".format(len(smiles_list) + 1))
        print("---------------------------------")
        i = i+1
    print("========================================")

    print(len(smiles_list))
    df = pd.DataFrame({'smiles_list' : smiles_list})

    with pd.ExcelWriter(file_path, engine='openpyxl', mode='a') as writer:
        # 如果文件不存在，mode='w' 会创建一个新文件，mode='a' 表示追加模式
        # 将 DataFrame 写入名为 'Sheet3' 的工作表，如果工作表已存在，将覆盖它
        df.to_excel(writer, sheet_name="smiles", index=False)





if __name__ == '__main__':
    file_path = "/data/temp/HOV.xlsx"

    getSmilesFromCAS(file_path)