import os

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from mordred import Calculator, descriptors
from mordred.error import Missing




def unique_smiles(file_path, sheet_name, column_name):
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    # 步骤2: 去重
    # 假设我们要按照'column_name'这一列去重
    unique_df = df.drop_duplicates(subset=column_name, keep='first')

    # 步骤3: 写回文件
    with pd.ExcelWriter(file_path, engine='openpyxl', mode='a') as writer:
        # 如果文件不存在，mode='w' 会创建一个新文件，mode='a' 表示追加模式
        # 将 DataFrame 写入名为 'Sheet3' 的工作表，如果工作表已存在，将覆盖它
        unique_df.to_excel(writer, sheet_name='unique_smiles', index=False)





def getSmiles(file_path, sheet_name, column_name):
    # 读取Excel文件
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    # 访问列
    smiles_list = df[column_name]
    return smiles_list.values

'''
批量获取共同有效的描述符写入文件
'''
def get_valid_descriptors(file_path, list):

    # mordred一共提供1826种描述符
    index = np.zeros(1826)
    for smiles in list:
        if smiles is None or smiles == "" or smiles == " ":
            continue
        # 读取分子结构
        molecule = Chem.MolFromSmiles(smiles)
        molecule = Chem.AddHs(molecule)

        # 为分子生成三维构象
        if molecule.GetNumConformers() == 0:
            AllChem.EmbedMolecule(molecule, AllChem.ETKDG())
            AllChem.UFFOptimizeMolecule(molecule)

        # 检查分子是否有效
        if molecule is None or molecule.GetNumAtoms() == 0:
            raise ValueError("无效的分子结构")

        # 确保分子有三维坐标
        if molecule.GetNumConformers() > 0:
            calculator = Calculator(descriptors, ignore_3D=False)
            results = calculator(molecule)
            i = 0
            for desc, value in results.items():
                # 去除没有的描述符
                if type(value) is Missing:
                    index[i] = 1
                i += 1
        else:
            print('无法生成分子的三维构象')

    # 统计共同存在的描述符
    non_zero_elements = index != 1
    print(np.sum(non_zero_elements))

    # 写入共同有效的描述符
    molecule = Chem.MolFromSmiles("C")
    calculator = Calculator(descriptors, ignore_3D=False)
    results = calculator(molecule)
    i = 0
    valid_descriptors = []
    for desc, value in results.items():
        if index[i] == 0:
            valid_descriptors.append(desc)
        i += 1
    print("共同有效描述符：", len(valid_descriptors))

    df = pd.DataFrame({'valid_descriptors' : valid_descriptors})
    with pd.ExcelWriter(file_path, engine='openpyxl', mode='a') as writer:
        # 如果文件不存在，mode='w' 会创建一个新文件，mode='a' 表示追加模式
        # 将 DataFrame 写入名为 'Sheet3' 的工作表，如果工作表已存在，将覆盖它
        df.to_excel(writer, sheet_name='valid_descriptors', index=False)

    return valid_descriptors, len(valid_descriptors)

'''
批量获取所有化合物的有效的描述符的值写入文件
'''

def saveDescriptorsValues(file_path, valid_descriptors, smiles_list):

    hovDescriptorsValues = []
    i = 1
    for smiles in smiles_list:
        if smiles is None or smiles == "" or smiles == " ":
            hovDescriptorsValues.append([])
            i += 1
            continue
        print(i)
        # 读取分子结构
        molecule = Chem.MolFromSmiles(smiles)
        molecule = Chem.AddHs(molecule)

        # 为分子生成三维构象
        if molecule.GetNumConformers() == 0:
            AllChem.EmbedMolecule(molecule, AllChem.ETKDG())
            AllChem.UFFOptimizeMolecule(molecule)

        # 检查分子是否有效
        if molecule is None or molecule.GetNumAtoms() == 0:
            raise ValueError("无效的分子结构")

        # 确保分子有三维坐标
        if molecule.GetNumConformers() > 0:
            calculator = Calculator(descriptors, ignore_3D=False)
            results = calculator(molecule)
            mol_values = []
            # 保存该化合物的有效描述符的值
            for key in valid_descriptors:
                mol_values.append(results[key])
            hovDescriptorsValues.append(mol_values)
        else:
            print('无法生成分子的三维构象')
        i = i + 1
    print(len(hovDescriptorsValues))
    df = pd.DataFrame(hovDescriptorsValues, columns=valid_descriptors)

    with pd.ExcelWriter(file_path, engine='openpyxl', mode='a') as writer:
        # 如果文件不存在，mode='w' 会创建一个新文件，mode='a' 表示追加模式
        # 将 DataFrame 写入名为 'Sheet3' 的工作表，如果工作表已存在，将覆盖它
        df.to_excel(writer, sheet_name='Sheet2', index=False)




if __name__ == '__main__':


    # 步骤2: 设置data文件夹的路径
    data_folder_path = '../data/temp'
    smiles_sheet_name = "Sheet1"
    smiles_column_name = "smiles_list"
    info_list = []
    for filename in os.listdir(data_folder_path):
        # 构建完整的文件路径
        file_path = os.path.join(data_folder_path, filename)
        print(file_path)
        # 是否已经处理过
        df = pd.read_excel(file_path, sheet_name=None)
        sheet_name_to_check = 'unique_smiles'  # 替换为你想要检查的sheet名称
        if sheet_name_to_check in df.keys():
            continue

        # smiles去重
        unique_smiles(file_path, smiles_sheet_name, smiles_column_name)
        # 获取smiles
        smiles_list = getSmiles(file_path, "unique_smiles", smiles_column_name)
        # 获取共同有效描述符
        valid_descriptors, descriptors_length = get_valid_descriptors(file_path, smiles_list)
        # 保存描述符信息
        saveDescriptorsValues(file_path, valid_descriptors, smiles_list)

        info = file_path + ", data_length: " + str(len(smiles_list)) + ", descriptors_length: " + str(descriptors_length)
        print(info)
        info_list.append(info)
    print(info_list)

