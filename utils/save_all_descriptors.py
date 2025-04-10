
import pickle

import pandas as pd
from mordred import Calculator, descriptors
from mordred._base import descriptor
from mordred.error import Missing
from rdkit import Chem
from rdkit.Chem import AllChem


def calculate_descriptors(smiles_list):

    result = {}
    i = 1
    for smiles in smiles_list:
        print(i)
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


        if molecule.GetNumConformers() > 0:
            calculator = Calculator(descriptors, ignore_3D=False)
            results = calculator(molecule)
            mol_descriptors = {}
            for key, value in results.items():
                str_repr = str(key)  # 示例输出: '<Descriptor: ABCIndex>'
                name = str_repr.split(": ")[-1].strip(">")  # 得到 'ABCIndex
                if type(value) is Missing:
                    mol_descriptors[name] = 0
                else:
                    mol_descriptors[name] = value
            result[smiles] = mol_descriptors
        i += 1

    print("描述符计算完成。")
    print("描述符字典：", len(result.keys()))

    with open("../result/all_descriptors.pkl", "wb") as f:
        pickle.dump({"descriptors_dict": result}, f)



if __name__ == '__main__':
    all_data_file = "C:\\Users\\lx\\Desktop\\网站构建信息\\所有预测数据.xlsx"
    all_df = pd.read_excel(all_data_file, sheet_name="all_data")
    smiles_list = all_df["SMILES"].tolist()



    calculate_descriptors(smiles_list)
    # calculate_descriptors(["C"])


