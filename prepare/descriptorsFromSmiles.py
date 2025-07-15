import pandas as pd

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from mordred import Calculator, descriptors
from mordred.error import Missing

def getSmiles(file_path, sheet_name, column_name):

    df = pd.read_excel(file_path, sheet_name=sheet_name)
    smiles_list = df[column_name]
    return smiles_list.values


def get_valid_descriptors(list, file_name):


    index = np.zeros(1826)
    for smiles in list:
        molecule = Chem.MolFromSmiles(smiles)
        molecule = Chem.AddHs(molecule)
        if molecule.GetNumConformers() == 0:
            AllChem.EmbedMolecule(molecule, AllChem.ETKDG())
            AllChem.UFFOptimizeMolecule(molecule)

        if molecule is None or molecule.GetNumAtoms() == 0:
            raise ValueError("Invalid molecular structure")

        if molecule.GetNumConformers() > 0:
            calculator = Calculator(descriptors, ignore_3D=False)
            results = calculator(molecule)
            i = 0
            for desc, value in results.items():
                if type(value) is Missing:
                    index[i] = 1
                i += 1
        else:
            print('Unable to generate three-dimensional conformations of molecules')

    non_zero_elements = index != 1
    print(np.sum(non_zero_elements))

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


def saveDescriptorsValues(descriptors_file_path, descriptors_sheet_name, descriptors_column_name,
                          smiles_list, output_file_name):

    df = pd.read_excel(descriptors_file_path, sheet_name=descriptors_sheet_name)

    valid_descriptors = df[descriptors_column_name]
    hovDescriptorsValues = []
    i = 1
    for smiles in smiles_list:
        print(i)

        molecule = Chem.MolFromSmiles(smiles)
        molecule = Chem.AddHs(molecule)


        if molecule.GetNumConformers() == 0:
            AllChem.EmbedMolecule(molecule, AllChem.ETKDG())
            AllChem.UFFOptimizeMolecule(molecule)

        if molecule is None or molecule.GetNumAtoms() == 0:
            raise ValueError("Invalid molecular structure")

        if molecule.GetNumConformers() > 0:
            calculator = Calculator(descriptors, ignore_3D=False)
            results = calculator(molecule)
            mol_values = []

            for key in valid_descriptors:
                mol_values.append(results[key])
            hovDescriptorsValues.append(mol_values)
        else:
            print('Unable to generate three-dimensional conformations of molecules')
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
