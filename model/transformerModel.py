
import torch.nn as nn
import torch.nn.functional as F


class CustomTransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=F.relu,
                 layer_norm_eps=1e-5, batch_first=True, norm_first=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = activation

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super().__setstate__(state)

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False, return_attention=False):

        x = src
        if self.norm_first:
            attn_output, attn_weights = self._sa_block(self.norm1(x), src_mask, src_key_padding_mask, is_causal)
            x = x + attn_output
            x = x + self._ff_block(self.norm2(x))
        else:
            attn_output, attn_weights = self._sa_block(x, src_mask, src_key_padding_mask, is_causal)
            x = self.norm1(x + attn_output)
            x = self.norm2(x + self._ff_block(x))

        if return_attention:
            return x, attn_weights
        else:
            return x

    def _sa_block(self, x, attn_mask, key_padding_mask, is_causal=False):
        attn_output, attn_weights = self.self_attn(x, x, x,
                                                   attn_mask=attn_mask,
                                                   key_padding_mask=key_padding_mask,
                                                   is_causal=is_causal,
                                                   need_weights=True,
                                                   average_attn_weights=False)
        return self.dropout1(attn_output), attn_weights

    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class SimpleTransformerRegressor(nn.Module):

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

        self.encoder_layer = CustomTransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            dropout=dropout_rate
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

    def forward(self, x, return_attention=False):
        if return_attention:

            attention_weights = []
            layer_input = x

            for layer in self.transformer_encoder.layers:

                layer_output, layer_attn = layer(layer_input, return_attention=True)
                attention_weights.append(layer_attn)
                layer_input = layer_output

            x = layer_input

            x = F.leaky_relu(self.fc1(x))
            x = self.dropout1(x)
            x = x.reshape(x.shape[0], -1)
            x = F.leaky_relu(self.fc2(x))
            x = self.dropout2(x)
            x = F.leaky_relu(self.fc3(x))
            x = self.dropout3(x)
            x = self.fc4(x)

            return x, attention_weights
        else:

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

