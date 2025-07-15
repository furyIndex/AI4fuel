import json

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from common.parse_args import args
from descriptors_group.getGCNDescriptors import splitTrainTest, knn_train, l1_norm, norm_adj_train, knn_val, \
    norm_adj_val
from load_model.loadGCNModel import load_model

device = torch.device(args.device)






def save_learned_gcn_data(file_path, sheet_name, model, k, a, grouped=True):

    x_train, x_test, labels, descriptor_name = splitTrainTest(file_path)



    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    x_full = np.vstack((x_train, x_test))
    train_length = len(x_train)
    test_length = len(x_test)
    full_length = len(x_full)


    train_input = torch.tensor(x_train).to(device)
    test_input = torch.tensor(x_test).to(device)
    train_embedding = model(train_input)
    test_embedding = model(test_input)
    train_embedding = train_embedding.cpu().detach().numpy()
    test_embedding = test_embedding.cpu().detach().numpy()


    train_similarity_matrix = np.dot(train_embedding, train_embedding.T)
    test_similarity_matrix = np.dot(test_embedding, train_embedding.T)


    train_knn_matrix = knn_train(train_similarity_matrix, k)
    train_norm_matrix = l1_norm(train_knn_matrix)
    train_A, train_degree = norm_adj_train(train_norm_matrix, a)
    train_gcn_matrix = np.dot(train_A, x_train)


    test_block = knn_val(test_similarity_matrix, test_length, full_length, k)
    train_block = np.hstack([train_A, np.zeros((train_length, test_length))])
    full_adj = np.vstack([train_block, test_block])


    full_A = norm_adj_val(full_adj, train_degree, a)
    full_matrix = np.dot(full_A, x_full)
    test_gcn_matrix = full_matrix[-test_length:]


    gcn_matrix = np.vstack((train_gcn_matrix, test_gcn_matrix))
    pre_group_df = pd.DataFrame(gcn_matrix, columns=descriptor_name)

    result_df = pd.DataFrame()
    if grouped:
        max_columns = max(len(descriptors) for descriptors in descriptorsMapping.values())
        for type, descriptors in descriptorsMapping.items():
            exist_df = [descriptor for descriptor in descriptors if descriptor in pre_group_df.columns]
            if exist_df:
                group_df = pre_group_df[exist_df]
                padded_values = group_df.apply(lambda x: x.tolist() + [0] * (max_columns - len(x)), axis=1)
                result_df[type] = padded_values
    else:
        result_df = pre_group_df

    result_df.insert(loc=0, column='value', value=labels)

    with pd.ExcelWriter(file_path, engine='openpyxl', mode='a') as writer:
        result_df.to_excel(writer, sheet_name=sheet_name, index=False)




if __name__ == '__main__':
    with open('./descriptorsMap/descriptorsMapping.json', 'r') as f:
        descriptorsMapping = json.load(f)


    data_dict = {
        '../data/other/HOV.xlsx' : "HOV_embedding_2_0.5871895242769624.pth",
        # '../data/other/LHV.xlsx' : "LHV_embedding_3_0.8772815444910687.pth",
        # '../data/other/RON.xlsx' : "RON_embedding_1_0.3760063468414003.pth",
        # '../data/other/MON.xlsx' : "MON_embedding_1_0.3775167319125256.pth",
        # '../data/other/CN.xlsx' : "CN_embedding_1_0.3480172106524897.pth",
        # '../data/other/YSI.xlsx' : "YSI_embedding_1_0.8152132600867458.pth",
        # '../data/other/Density.xlsx' : "Density_embedding_1_0.48608576201740794.pth",
        # '../data/other/TB.xlsx' : "TB_embedding_1_0.45683157386267037.pth",
        # '../data/other/TM.xlsx' : "TM_embedding_3_0.526635736508536.pth",
        # '../data/other/UFL.xlsx' : "UFL_embedding_3_0.6188434893620851.pth",
        # '../data/other/LFL.xlsx' : "LFL_embedding_3_0.7880042580593524.pth",
        # '../data/other/Viscosity.xlsx' : "Viscosity_embedding_1_0.7457670924015943.pth",
        # '../data/other/Enthalpy_of_Vaporization.xlsx' : "Enthalpy_of_Vaporization_embedding_3_0.8097393628062347.pth",
        # '../data/other/VP.xlsx' : "VP_embedding_1_0.833519575858319.pth",
        # '../data/other/DCN.xlsx' : "DCN_embedding_3_0.5978619025504474.pth",
        # '../data/other/Surface_tension.xlsx' : "Surface_tension_embedding_1_0.5136778129492438.pth",
        # '../data/other/Flash_point.xlsx' : "Flash_point_embedding_1_0.4566019803325665.pth"

    }

    model_folder = "../save_models_server/"

    for file_path, model_name in data_dict.items():
        print(file_path)
        model_path = model_folder + model_name

        embedding_net = load_model(model_path)
        embedding_net = embedding_net.to(device)

        save_learned_gcn_data(file_path, "lg_k=10_a=0.1", embedding_net, 10, 0.1, grouped=False)

