import os

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from mordred import Calculator, descriptors
from mordred.error import Missing




def unique_smiles(file_path, sheet_name, column_name):
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    unique_df = df.drop_duplicates(subset=column_name, keep='first')
    with pd.ExcelWriter(file_path, engine='openpyxl', mode='a') as writer:
        unique_df.to_excel(writer, sheet_name='unique_smiles', index=False)





def getSmiles(file_path, sheet_name, column_name):

    df = pd.read_excel(file_path, sheet_name=sheet_name)
    smiles_list = df[column_name]
    return smiles_list.values


def get_valid_descriptors(file_path, list):

    index = np.zeros(1826)
    for smiles in list:
        if smiles is None or smiles == "" or smiles == " ":
            continue

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
    print("Common effective descriptors：", len(valid_descriptors))
    df = pd.DataFrame({'valid_descriptors' : valid_descriptors})
    with pd.ExcelWriter(file_path, engine='openpyxl', mode='a') as writer:
        df.to_excel(writer, sheet_name='valid_descriptors', index=False)
    return valid_descriptors, len(valid_descriptors)



def saveDescriptorsValues(file_path, valid_descriptors, smiles_list):

    hovDescriptorsValues = []
    i = 1
    for smiles in smiles_list:
        if smiles is None or smiles == "" or smiles == " ":
            hovDescriptorsValues.append([])
            i += 1
            continue
        print("Processing the {}-th molecule".format(i))

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

    with pd.ExcelWriter(file_path, engine='openpyxl', mode='a') as writer:

        df.to_excel(writer, sheet_name='Sheet2', index=False)




if __name__ == '__main__':


    data_folder_path = '../data/ron_test'
    smiles_sheet_name = "Sheet1"
    smiles_column_name = "SMILES"
    info_list = []
    for filename in os.listdir(data_folder_path):

        file_path = os.path.join(data_folder_path, filename)
        print(file_path)
        df = pd.read_excel(file_path, sheet_name=None)
        sheet_name_to_check = 'unique_smiles'
        if sheet_name_to_check in df.keys():
            continue

        # unique_smiles(file_path, smiles_sheet_name, smiles_column_name)
        # smiles_list = getSmiles(file_path, "unique_smiles", smiles_column_name)

        smiles_list = getSmiles(file_path, smiles_sheet_name, smiles_column_name)
        valid_descriptors, descriptors_length = get_valid_descriptors(file_path, smiles_list)
        saveDescriptorsValues(file_path, valid_descriptors, smiles_list)
        info = file_path + ", data_length: " + str(len(smiles_list)) + ", descriptors_length: " + str(descriptors_length)
        print(info)
        info_list.append(info)
    print(info_list)

