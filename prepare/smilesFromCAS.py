import pubchempy as pcp
import pandas as pd


def getData():
    # 读取Excel文件
    df = pd.read_excel('D:\\code\\PyCharm_WorkSpace\\ai4fuel\\data\\LHV_unique.xlsx', sheet_name='Sheet1')
    smiles_list = []
    # 访问列
    name_data = df['Name']
    cas_data = df['CAS']
    print(len(name_data))
    print(len(cas_data))
    j = 2
    for i in cas_data.values:
        print("当前索引：{}".format(j))
        print(type(i))
        print(i)
        j += 1
        # if type(i) is str:
        #     j += 1
        #     print(i)
    print(j)




'''
获取每一个化合物的smiles表达式
'''
def getSmilesFromCAS():
    # 读取Excel文件
    df = pd.read_excel('D:\\code\\PyCharm_WorkSpace\\ai4fuel\\data\\LHV_unique.xlsx', sheet_name='Sheet1')
    smiles_list = []
    # 访问列
    name_data = df['Name']
    cas_data = df['CAS']
    cas_list = []
    # 去除读取的excel中不规范的cas值（比如有的读出来是0）
    for cas in cas_data.values:
        if type(cas) is str:
            cas_list.append(cas)
    print(len(cas_list))
    i = 2
    for cas in cas_list:
        print(i)
        print(cas)
        # try:
        compounds = pcp.get_compounds(cas, 'name')
        # except (IndexError, ValueError) as e:
        #     print(f"CAS号 {cas} 无法找到对应的化合物信息。错误信息：{e}")
        #     smiles_list.append("null")
        #     i += 1
        #     continue
        # 如果根据cas查出来的化合物有多种，就根据名字查
        # if len(compounds) > 1:
        #     compounds = pcp.get_compounds(name_data.values[i], 'name')
        #
        # print(len(compounds))
        # print(compounds)
        print(len(compounds))
        if len(compounds) == 1:
            compound = compounds[0]
            print(compound.isomeric_smiles)
            # 如果单个化合物的smiles为空，就填空
            if compound.isomeric_smiles is None:
                smiles_list.append("null")
            else:
                smiles_list.append(compound.isomeric_smiles)
        if len(compounds) == 0:
            smiles_list.append("null")
        # 如果查出来的化合物仍然有多种，就把他们的smiles拼一起
        if len(compounds) > 1:
            smiles_list.append("multiple")
        print("列表长度：{}".format(len(smiles_list) + 1))
        print("---------------------------------")
        i = i+1
    print("========================================")

    print(len(smiles_list))
    df = pd.DataFrame({'smiles_list' : smiles_list})
    df.to_excel('smiles.xlsx', index=False)



if __name__ == '__main__':
    getSmilesFromCAS()
    # getData()