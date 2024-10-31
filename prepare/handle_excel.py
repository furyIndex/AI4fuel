import pubchempy as pcp
import pandas as pd


'''
获取每一个化合物的smiles表达式
'''
def smilesFromName():
    # 读取Excel文件
    df = pd.read_excel('D:\\code\\PyCharm_WorkSpace\\ai4fuel\\data\\HoV_unique.xlsx', sheet_name='Sheet1')
    name_list = []
    smiles_list = []
    hov_list = []
    # 访问列
    name_data = df['Name']
    value_data = df['HOV']
    info = dict(zip(name_data, value_data))


    i = 0
    for name in name_data.values:
        if i % 10 == 0:
            print(i)
        i += 1
        compounds = pcp.get_compounds(name, 'name')

        if len(compounds) == 0:
            print(name)
            name_list.append(name)
            smiles_list.append(" ")
            hov_list.append(info[name])
            continue
        for compound in compounds:
            if compound.canonical_smiles is None:
                name_list.append(name)
                smiles_list.append(" ")
                hov_list.append(info[name])
                continue
            print(compound.canonical_smiles)
            name_list.append(name)
            smiles_list.append(compound.canonical_smiles)
            hov_list.append(info[name])
        print("------------------")
    print("========================================")
    print(len(name_list))
    print(len(smiles_list))
    print(len(hov_list))
    df = pd.DataFrame({'name_list' : name_list, 'smiles_list' : smiles_list, 'hov_list' : hov_list})
    df.to_excel('smiles.xlsx', index=False)


def getHoV():
    df = pd.read_excel('D:\\code\\PyCharm_WorkSpace\\ai4fuel\\data\\Fuel_data.xlsx', sheet_name='Sheet1')
    result = []
    # 访问列
    column_data = df['HoV(kJ/kg)']
    return column_data.values


if __name__ == '__main__':
    # lables = getHoV()
    # print(type(lables))
    # print(lables)
    smilesFromName()
    # print(len(smiles))