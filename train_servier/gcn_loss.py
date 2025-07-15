import torch
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader
from dataset import molDataset
from load_model.loadGCNModel import predict_data, mean_squared_error, split_LG_Data
from model.transformerModel import SimpleTransformerRegressor

from common.parse_args import args


def showFig(data, name, color, save_path):

    epochs = list(data.keys())
    losses = list(data.values())

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, losses, color=color, label='{} Loss'.format(name), linewidth=3)

    plt.title('{} Loss Over Epochs'.format(name))
    plt.xlabel('Epoch', fontsize=16)
    plt.ylabel('Loss', fontsize=16)

    plt.legend(
        fontsize=20,
        handlelength=3,
        markerscale=1.5
    )
    plt.yscale('log')
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches='tight')


    plt.show()
    plt.close()




def showTwoFig(train_data, test_data):

    train_epochs = list(train_data.keys())
    train_losses = list(train_data.values())
    test_epochs = list(test_data.keys())
    test_losses = list(test_data.values())

    plt.figure(figsize=(10, 6))
    plt.plot(train_epochs, train_losses, marker='o', label='Training Loss')
    plt.plot(test_epochs, test_losses, marker='s', label='Testing Loss')

    plt.title('Training and Testing Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.legend()

    plt.grid(True)
    plt.show()



# MSLE Loss
def msle_loss(pred, target):
    pred = torch.clamp(pred, min=1e-7)
    target = torch.clamp(target, min=1e-7)
    log_pred = torch.log1p(pred)
    log_target = torch.log1p(target)
    loss = torch.mean((log_pred - log_target) ** 2)
    return loss







if __name__ == '__main__':



    device = torch.device(args.device)


    data_name = "Density"

    vp_list = [
        'lg_k=10_a=0.1',
        # 'lg_k=10_a=0.1',
        # 'new_lg_k=10_a=0.1',
        # 'k=10_a=0.1',
        # 'k=10_a=0.2',
        # 'k=10_a=0.5',
        # 'k=50_a=0.1',
        # 'k=50_a=0.2',
        # 'k=50_a=0.5',
    ]


    for sheet_name in vp_list:
        print(sheet_name)

        bestModel = None
        maxR2 = 0
        best_k = 0

        file_path = '../data/other/{}.xlsx'.format(data_name)



        x_train, x_test, y_train, y_test = split_LG_Data(file_path, sheet_name)
        print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

        best_train = None
        best_test = None


        for k in range(1):

            train_record = {}
            test_record = {}

            # 初始化dataset
            train_dataset = molDataset.MolDataset(x_train, y_train)
            test_dataset = molDataset.MolDataset(x_test, y_test)


            train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

            train_length = len(train_dataset)

            transformModel = SimpleTransformerRegressor(input_dim=args.input_dim,
                                                        seq_length=49,
                                                        dim_feedforward=1024,
                                                        num_heads=2,
                                                        num_layers=1,
                                                        hidden_dim_1=64,
                                                        hidden_dim_2=1024,
                                                        hidden_dim_3=128,
                                                        dropout_rate=0.5)


            transformModel = transformModel.to(device)
            optimizer = torch.optim.AdamW(transformModel.parameters(), lr=0.001198660111377494)

            loss_function = torch.nn.MSELoss()
            loss_function = loss_function.to(device)


            num_epochs = 297
            total_train_step = 0
            total_eval_step = 0


            for epoch in range(num_epochs):
                transformModel.train()
                train_loss = 0
                for data, target in train_loader:
                    data, target = data.to(device), target.to(device)
                    optimizer.zero_grad()
                    output = transformModel(data)

                    # item_loss = msle_loss(output, target)
                    item_loss = loss_function(output.float(), target.float())

                    train_loss += item_loss.item()
                    item_loss.backward()
                    optimizer.step()
                    total_train_step += 1
                train_loss /= train_length

                if epoch % 2 == 0:
                    train_record[epoch] = train_loss

                transformModel.eval()
                with torch.no_grad():
                    y_true, y_pred = predict_data(x_test, y_test, transformModel)
                    mse = mean_squared_error(y_true, y_pred)
                    r2 = r2_score(y_test, y_pred)
                if epoch % 2 == 0:
                    test_record[epoch] = mse

            transformModel.eval()
            with torch.no_grad():
                y_ture, y_pred = predict_data(x_test, y_test, transformModel)
                r2 = r2_score(y_test, y_pred)
                if r2 > maxR2:
                    maxR2 = r2
                    bestModel = transformModel
                    best_k = k
                    best_train = train_record
                    best_test = test_record

            k = k + 1



        print(best_train)
        print(best_test)

        save_path_1 = "D:\\code\\PyCharm_WorkSpace\\ai4fuel\\result_png\\{}_train.png".format(data_name)
        save_path_2 = "D:\\code\\PyCharm_WorkSpace\\ai4fuel\\result_png\\{}_test.png".format(data_name)

        showFig(best_train, "Training", color="#3BA8B7", save_path=save_path_1)
        showFig(best_test, "Testing", color="#B25840", save_path=save_path_2)


        print("===============================================")


        # torch.save(bestModel, '../save_models_server/VP_{}_{}.pth'.format (sheet_name, maxR2))




