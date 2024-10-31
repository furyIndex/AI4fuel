from prepare.handle_excel import *
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from mordred import Calculator, descriptors
from mordred.error import Missing

def getSmiles(file_path, sheet_name, column_name):
    # 读取Excel文件
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    # 访问列
    smiles_list = df[column_name]
    return smiles_list.values

'''
批量获取共同有效的描述符写入文件
'''
def get_valid_descriptors(list, file_name):

    # mordred一共提供1826种描述符
    index = np.zeros(1826)
    for smiles in list:
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
    df = pd.DataFrame({'valid_descriptors' : valid_descriptors})
    df.to_excel(file_name, index=False)

'''
批量获取所有化合物的有效的描述符的值写入文件
'''
def saveDescriptorsValues(descriptors_file_path, descriptors_sheet_name, descriptors_column_name,
                          smiles_list, output_file_name):
    # 读取Excel文件
    df = pd.read_excel(descriptors_file_path, sheet_name=descriptors_sheet_name)
    # 访问列
    valid_descriptors = df[descriptors_column_name]
    hovDescriptorsValues = []
    i = 1
    for smiles in smiles_list:
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
    df.to_excel(output_file_name, index=False)




if __name__ == '__main__':

    smiles_file_path = "D:\\code\\PyCharm_WorkSpace\\ai4fuel\\data\\RON-MON-CN.xlsx"
    smiles_sheet_name = "CN"
    smiles_column_name = "SMILES"

    descriptors_file_path = "D:\\code\\PyCharm_WorkSpace\\ai4fuel\\data\\CN_valid_descriptors.xlsx"
    descriptors_sheet_name = "Sheet1"
    descriptors_column_name = "valid_descriptors"
    smiles_list = getSmiles(smiles_file_path, smiles_sheet_name, smiles_column_name)
    # get_valid_descriptors(smiles_list, "MON_valid_descriptors.xlsx")
    saveDescriptorsValues(descriptors_file_path, descriptors_sheet_name, descriptors_column_name,
                          smiles_list, "CN_descriptors_values.xlsx")
