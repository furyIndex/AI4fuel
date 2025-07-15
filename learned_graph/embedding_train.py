import os
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from common.parse_args import args
from dataset import molDataset
from model.embedding_net import EmbeddingMLP, ContrastiveLoss, embedding_val_func
import numpy as np
from train_servier.gcn_loss import showFig



device = torch.device(args.device)




def split_descriptors_data(file_path):
    df = pd.read_excel(file_path, sheet_name='Sheet2')
    grouped = df.groupby(df.columns[0])
    pre_data_len = len(df.columns)
    train_set = np.empty((0, pre_data_len))
    test_set = np.empty((0, pre_data_len))
    for name, group in grouped:
        group_data = group.values
        group_length = group_data.shape[0]

        np.random.seed(42)
        np.random.shuffle(group_data)

        split = int(group_length * 0.8)
        train_data = group_data[:split, :]
        test_data = group_data[split:, :]

        train_set = np.concatenate((train_set, train_data), axis=0)
        test_set = np.concatenate((test_set, test_data), axis=0)

    full_data = np.vstack((train_set, test_set))
    split = int(full_data.shape[0] * 0.8)
    x_train = full_data[:split, 2:].astype(np.float32)
    x_test = full_data[split:, 2:].astype(np.float32)
    y_train = full_data[:split, 1:2].astype(np.float32)
    y_test = full_data[split:, 1:2].astype(np.float32)

    x_scaler = StandardScaler()
    x_train = x_scaler.fit_transform(x_train)
    x_test = x_scaler.transform(x_test)


    y_scaler = StandardScaler()
    y_train = y_scaler.fit_transform(y_train)
    y_test = y_scaler.transform(y_test)

    return x_train, x_test, y_train, y_test




def get_add_split_data(file_path):

    full_data = pd.read_excel(file_path, sheet_name='split_data')


    full_data = full_data.values

    split = int(full_data.shape[0] * 0.8)
    x_train = full_data[:split, 3:].astype(np.float32)
    x_test = full_data[split:, 3:].astype(np.float32)
    y_train = full_data[:split, 2:3].astype(np.float32)
    y_test = full_data[split:, 2:3].astype(np.float32)

    x_scaler = StandardScaler()
    x_train = x_scaler.fit_transform(x_train)
    x_test = x_scaler.transform(x_test)


    y_scaler = StandardScaler()
    y_train = y_scaler.fit_transform(y_train)
    y_test = y_scaler.transform(y_test)

    return x_train, x_test, y_train, y_test







def train_embedding():

    train_dataset = molDataset.MolDataset(x_train, y_train)
    train_length = len(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    embedding_net = EmbeddingMLP(input_dim, 512, 128)
    embedding_net = embedding_net.to(device)
    optimizer = torch.optim.AdamW(embedding_net.parameters(), lr=0.01, weight_decay=1e-3)


    loss_function = ContrastiveLoss(margin=margin)
    loss_function = loss_function.to(device)


    num_epochs = 50


    train_record = {}
    test_record = {}


    for epoch in range(num_epochs):
        embedding_net.train()
        train_loss = 0
        for feature, target in train_loader:
            feature, target = feature.to(device), target.to(device)
            optimizer.zero_grad()
            embeddings = embedding_net(feature)

            item_loss = loss_function.forward(embeddings, target)

            train_loss += item_loss.item()
            item_loss.backward()
            optimizer.step()
        train_loss /= train_length
        train_record[epoch] = train_loss
        print("Epoch: {}, train loss：{}.".format(epoch, train_loss))

        embedding_net.eval()
        with torch.no_grad():
            test_input = torch.tensor(x_test).float().to(device)
            test_label = torch.tensor(y_test).float().to(device)
            pred = embedding_net(test_input)
            eval_loss = loss_function.forward(pred, test_label).cpu().detach()
            print("Epoch: {}, test loss：{}.".format(epoch, eval_loss))
            test_record[epoch] = eval_loss

    embedding_net.eval()
    with torch.no_grad():
        test_input = torch.tensor(x_test).float().to(device)
        test_label = torch.tensor(y_test).float().to(device)
        pred = embedding_net(test_input)
        spearmanr_res = embedding_val_func(pred, test_label)
        print("spearman’s rank correlation coefficient：", spearmanr_res)



    return embedding_net, spearmanr_res[0], train_record, test_record


if __name__ == '__main__':

    margin = 3.0
    data_dict = {
        '../data/ron_test/add_data.xlsx' : 1393,
        # '../data/other/HOV.xlsx' : 1288,
        # '../data/other/LHV.xlsx' : 1287,
        # '../data/other/RON.xlsx' : 1393,
        # '../data/other/MON.xlsx' : 1394,
        # '../data/other/CN.xlsx' : 1394,
        # '../data/other/YSI.xlsx' : 1394,
        # '../data/other/Density.xlsx' : 1348,
        # '../data/other/TB.xlsx' : 1287,
        # '../data/other/TM.xlsx' : 1287,
        # '../data/other/UFL.xlsx' : 1288,
        # '../data/other/LFL.xlsx' : 1287,
        # '../data/other/Viscosity.xlsx' : 1288,
        # '../data/other/Enthalpy_of_Vaporization.xlsx' : 1288,
        # '../data/other/VP.xlsx' : 1287,
        # '../data/other/DCN.xlsx' : 1453,
        # '../data/other/Surface_tension.xlsx' : 1348,
        # '../data/other/Flash_point.xlsx' : 1287
    }

    optimal_results = []

    for file_path, input_dim in data_dict.items():

        file_name_with_extension = os.path.basename(file_path)
        file_name = os.path.splitext(file_name_with_extension)[0]
        print(file_name)
        bestModel = None
        maxRes = 0
        train_best = None
        test_best = None


        x_train, x_test, y_train, y_test = split_descriptors_data(file_path)
        # x_train, x_test, y_train, y_test = get_add_split_data(file_path)


        for k in range(10):
            embedding_net, res, train_record, test_record = train_embedding()
            if res > maxRes:
                maxRes = res
                bestModel = embedding_net
                train_best = train_record
                test_best = test_record


        print("best spearman’s rank correlation coefficient：", maxRes)

        save_path_1 = "../result_png/{}_train.png".format(file_name)
        save_path_2 = "..//result_png/{}_test.png".format(file_name)
        showFig(train_best, "Training", color="navy", save_path=save_path_1)
        showFig(test_best, "Testing", color="red", save_path=save_path_2)


        torch.save(bestModel, '../save_models_server/add_ron_embedding_{}_{}.pth'.format(file_name, int(margin), maxRes))


        test_input = torch.tensor(x_test).float().to(device)
        test_label = torch.tensor(y_test).float().to(device)
        spearmanr_res = embedding_val_func(test_input, test_label)
        print("original spearman’s rank correlation coefficient：", spearmanr_res)
