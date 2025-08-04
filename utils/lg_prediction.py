import json
import pickle

import numpy as np
import pandas as pd
import torch
from mordred import Calculator, descriptors
from mordred.error import Missing
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.preprocessing import StandardScaler

from common.parse_args import args
from descriptors_group.getGCNDescriptors import knn_val, norm_adj_val
from load_model.load_model_utils import load_model

device = torch.device(args.device)





def load_x_train(file_path):

    with open(file_path, "rb") as f:
        loaded_data = pickle.load(f)

    x_train = loaded_data["x_train"]
    descriptor_name = loaded_data["descriptor_name"]

    print("Load data x_train：", x_train.shape)
    print("Number of data descriptors loaded：", len(descriptor_name))
    return x_train, descriptor_name


def load_train_A(file_path):
    with open(file_path, "rb") as f:
        loaded_data = pickle.load(f)
    train_A = loaded_data["train_A"]
    train_degree = loaded_data["train_degree"]

    print("Load data x_train: ", train_A.shape)
    print("Load data train_degree: ", train_degree.shape)
    return train_A, train_degree

def load_train_embedding(file_path):
    with open(file_path, "rb") as f:
        loaded_data = pickle.load(f)
    train_embedding = loaded_data["train_embedding"]
    print("load data train_embedding: ", train_embedding.shape)
    return train_embedding





def calculate_descriptors(file_path, smiles_list, descriptor_name):

    result = []
    for smiles in smiles_list:
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
            mol_values = []
            for key in descriptor_name:
                desc_value = results[key]
                if type(desc_value) is Missing:
                    print(key, "  Missing")
                    mol_values.append(0)
                else:
                    mol_values.append(desc_value)
            result.append(mol_values)

    result = np.array(result)
    print("Descriptor calculation completed。")
    return result



def get_descriptors_from_pkl(smiles_list, descriptor_name):

    with open("../result/all_descriptors.pkl", "rb") as f:
        loaded_data = pickle.load(f)
    smiles_descriptors = loaded_data["descriptors_dict"]

    result = []
    i = 1
    print("================Get descriptors==================")
    for smiles in smiles_list:
        print(i)
        mol_descriptors = []

        descriptors_dict = smiles_descriptors[smiles]

        for descriptor in descriptor_name:
            mol_descriptors.append(descriptors_dict[descriptor])
        result.append(mol_descriptors)
        i += 1

    result = np.array(result)
    return result






def get_lg_gcn_embedding(x_train, train_embedding,  train_A, train_degree, x_test, model):


    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    x_full = np.vstack((x_train, x_test))
    train_length = len(x_train)
    test_length = len(x_test)
    full_length = len(x_full)

    test_input = torch.tensor(x_test).float().to(device)
    test_embedding = model(test_input)
    test_embedding = test_embedding.cpu().detach().numpy()

    test_similarity_matrix = np.dot(test_embedding, train_embedding.T)

    test_block = knn_val(test_similarity_matrix, test_length, full_length, 10)

    train_block = np.hstack([train_A, np.zeros((train_length, test_length))])
    full_adj = np.vstack([train_block, test_block])


    full_A = norm_adj_val(full_adj, train_degree, 0.1)
    full_matrix = np.dot(full_A, x_full)
    test_gcn_matrix = full_matrix[-test_length:]


    pre_group_df = pd.DataFrame(test_gcn_matrix, columns=descriptor_name)

    max_columns = max(len(descriptors) for descriptors in descriptorsMapping.values())
    result_df = pd.DataFrame()


    for type, descriptors in descriptorsMapping.items():
        exist_df = [descriptor for descriptor in descriptors if descriptor in pre_group_df.columns]
        if exist_df:
            group_df = pre_group_df[exist_df]
            padded_values = group_df.apply(lambda x: x.tolist() + [0] * (max_columns - len(x)), axis=1)
            result_df[type] = padded_values

    tensor_data = df_to_tensor(result_df)
    tensor_data = tensor_data.to(device)

    print(tensor_data.shape)
    return tensor_data





def df_to_tensor(df):

    samples = []
    for _, row in df.iterrows():
        row_data = np.array(row.tolist())
        samples.append(row_data)
    data_np = np.stack(samples)

    tensor_data = torch.tensor(data_np, dtype=torch.float32)
    return tensor_data





if __name__ == '__main__':
    with open('../descriptors_group/descriptorsMap/descriptorsMapping.json', 'r') as f:
        descriptorsMapping = json.load(f)

    embedding_dict = {
        'HOV' : "HOV_embedding_2_0.5871895242769624.pth",
        'LHV' : "LHV_embedding_3_0.8772815444910687.pth",
        'RON' : "RON_embedding_1_0.3760063468414003.pth",
        'MON' : "MON_embedding_1_0.3775167319125256.pth",
        'CN' : "CN_embedding_1_0.3480172106524897.pth",
        'YSI' : "YSI_embedding_1_0.8152132600867458.pth",
        'Density' : "Density_embedding_1_0.48608576201740794.pth",
        'TB' : "TB_embedding_1_0.45683157386267037.pth",
        'TM' : "TM_embedding_3_0.526635736508536.pth",
        'UFL' : "UFL_embedding_3_0.6188434893620851.pth",
        'LFL' : "LFL_embedding_3_0.7880042580593524.pth",
        'Viscosity' : "Viscosity_embedding_1_0.7457670924015943.pth",
        'Enthalpy_of_Vaporization' : "Enthalpy_of_Vaporization_embedding_3_0.8097393628062347.pth",
        'VP' : "VP_embedding_1_0.833519575858319.pth",
        'DCN' : "DCN_embedding_3_0.5978619025504474.pth",
        'Surface_tension' : "Surface_tension_embedding_1_0.5136778129492438.pth",
        'Flash_point' : "Flash_point_embedding_1_0.4566019803325665.pth"

    }


    predict_model_dict = {
        'HOV':'HOV_lg_k=10_a=0.1_0.9162527918815613_model.pth',
        'LHV':'LHV_LG_lg_k=10_a=0.1_0.9952685236930847_model.pth',
        'RON':'RON_LG_lg_k=10_a=0.1_0.8893096446990967_model.pth',
        'MON':'MON_LG_lg_k=10_a=0.1_0.9218524098396301_model.pth',
        'CN':'CN_LG_lg_k=10_a=0.1_0.8978158235549927_model.pth',
        'YSI':'YSI_LG_lg_k=10_a=0.1_0.9909498691558838_model.pth',
        'Density':'Density_LG_lg_k=10_a=0.1_0.9421365261077881_model.pth',
        'TB':'TB_LG_lg_k=10_a=0.1_0.8605033159255981_model.pth',
        'TM':'TM_LG_lg_k=10_a=0.1_0.8670147061347961_model.pth',
        'UFL':'UFL_LG_lg_k=10_a=0.1_0.8814492225646973_model.pth',
        'LFL':'LFL_LG_lg_k=10_a=0.1_0.9601934552192688_model.pth',
        'Viscosity':'Viscosity_LG_lg_k=10_a=0.1_0.9803876280784607_model.pth',
        'Enthalpy_of_Vaporization':'Enthalpy_of_Vaporization_LG_lg_k=10_a=0.1_0.9861818552017212_model.pth',
        'VP':'VP_LG_lg_k=10_a=0.1_0.9621737599372864_model.pth',
        'DCN':'DCN_LG_lg_k=10_a=0.1_0.9323565363883972_model.pth',
        'Surface_tension':'Surface_tension_LG_lg_k=10_a=0.1_0.939228892326355_model.pth',
        'Flash_point':'Flash_point_LG_lg_k=10_a=0.1_0.9374099373817444_model.pth',

    }




    embedding_folder = "../save_models_server"
    predict_model_path = "../save_models_server"
    all_data_file = "../data/all_data/all_fuel_data.xlsx"
    all_df = pd.read_excel(all_data_file, sheet_name="all_data")

    for property, embedding_name in embedding_dict.items():
        print("======================={}=======================".format(property))

        embedding_path = embedding_folder + embedding_name
        embedding_model = load_model(embedding_path)
        embedding_model = embedding_model.to(device)


        x_train_path = "D:\\code\\PyCharm_WorkSpace\\ai4fuel\\result\\{}_train_data.pkl".format(property)
        train_A_path = "D:\\code\\PyCharm_WorkSpace\\ai4fuel\\result\\{}_train_A_data.pkl".format(property)
        train_embedding_path = "D:\\code\\PyCharm_WorkSpace\\ai4fuel\\result\\{}_train_embedding_data.pkl".format(property)
        x_train, descriptor_name = load_x_train(x_train_path)
        train_A, train_degree = load_train_A(train_A_path)
        train_embedding = load_train_embedding(train_embedding_path)

        smiles_list = all_df[all_df["{}_pre".format(property)].isna()]["SMILES"].tolist()
        print("Number of smiles to be predicted：", len(smiles_list))


        # test_descriptors = calculate_descriptors(all_data_file, smiles_list, descriptor_name)
        test_descriptors = get_descriptors_from_pkl(smiles_list, descriptor_name)


        input_data = get_lg_gcn_embedding(x_train, train_embedding, train_A, train_degree, test_descriptors, embedding_model)


        


        model = predict_model_dict[property]
        model_path = predict_model_path + model

        predict_model = load_model(model_path)
        predict_model = predict_model.to(device)
        prediction = predict_model(input_data)
        prediction = prediction.cpu().detach().numpy()
        prediction = np.squeeze(prediction)
        print("=======================Prediction completed===========================")
        print(prediction.shape)
        print(prediction)

        result_dict = dict(zip(smiles_list, prediction))
        all_df["{}_pre".format(property)] = all_df['SMILES'].map(result_dict)

    with pd.ExcelWriter(all_data_file, engine='openpyxl', mode='a') as writer:

        all_df.to_excel(writer, sheet_name="all_predict_data", index=False)