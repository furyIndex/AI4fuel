
import torch.nn as nn
import torch.nn.functional as F





class SimpleTransformerRegressor(nn.Module):
    '''
    :param input_dim:
    :param hidden_dim_1: 1-606
    :param hidden_dim_2:
    :param hidden_dim_3:
    '''
    def __init__(self,
                 seq_length,
                 input_dim,
                 dim_feedforward,
                 hidden_dim_1,
                 hidden_dim_2,
                 hidden_dim_3,
                 num_heads,
                 num_layers,
                 dropout_rate,
                 output_dim=1,
                 ):

        super(SimpleTransformerRegressor, self).__init__()


        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)


        self.fc1 = nn.Linear(input_dim, hidden_dim_1)
        self.fc2 = nn.Linear(seq_length * hidden_dim_1, hidden_dim_2)
        self.fc3 = nn.Linear(hidden_dim_2, hidden_dim_3)
        self.fc4 = nn.Linear(hidden_dim_3, output_dim)

        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.dropout3 = nn.Dropout(dropout_rate)

        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='leaky_relu')
        nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity='leaky_relu')
        nn.init.kaiming_uniform_(self.fc3.weight, nonlinearity='leaky_relu')
        nn.init.kaiming_uniform_(self.fc4.weight, nonlinearity='leaky_relu')



    def forward(self, x):

        x = self.transformer_encoder(x)

        x = F.leaky_relu(self.fc1(x))
        x = self.dropout1(x)
        x = x.reshape(x.shape[0], -1)

        x = F.leaky_relu(self.fc2(x))
        x = self.dropout2(x)

        x = F.leaky_relu(self.fc3(x))
        x = self.dropout3(x)

        x = self.fc4(x)

        return x


