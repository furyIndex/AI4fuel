import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns

from common.parse_args import args



device = torch.device(args.device)


def compute_attention_weights(model, input_data):
    input_data = torch.Tensor(input_data).to(device)

    # 定位 Transformer 层
    transformer_layer = model.encoder_layer  # 获取第一层 Transformer
    multihead_attention = transformer_layer.self_attn  # 获取 MultiheadAttention 层

    # 获取投影矩阵的形状
    embedding_dim = multihead_attention.in_proj_weight.shape[1]
    seq_len, batch_size, data_dim = input_data.shape  # input_data: [seq_len, batch_size, data_dim]

    # 获取查询、键、值的矩阵 Q, K, V
    Q = torch.matmul(input_data, multihead_attention.in_proj_weight[:embedding_dim, :].T) + multihead_attention.in_proj_bias[:embedding_dim]  # Q: [seq_len, batch_size, embedding_dim]
    K = torch.matmul(input_data, multihead_attention.in_proj_weight[embedding_dim:2 * embedding_dim, :].T) + multihead_attention.in_proj_bias[embedding_dim:2 * embedding_dim]  # K: [seq_len, batch_size, embedding_dim]
    V = torch.matmul(input_data, multihead_attention.in_proj_weight[2 * embedding_dim:, :].T) + multihead_attention.in_proj_bias[2 * embedding_dim:]  # V: [seq_len, batch_size, embedding_dim]

    # 计算 Q 和 K 的点积得到注意力得分
    attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (embedding_dim ** 0.5)  # [seq_len, batch_size, seq_len]

    # 应用 softmax 得到注意力权重
    attention_weights = F.softmax(attention_scores, dim=-1)  # [seq_len, batch_size, seq_len]

    return attention_weights

def get_attention_weights(model, input_data):
    input_data = torch.Tensor(input_data).to(device)

    output, attention_weights = model.encoder_layer.self_attn(input_data, input_data, input_data, need_weights=True)
    print("获取的注意力权重矩阵形状：", attention_weights.shape)
    average_attention_weights = torch.mean(attention_weights, dim=0)
    print(average_attention_weights.shape)
    return average_attention_weights


def showHeatMap(name, attention_weights, save_path):
    attention_matrix = attention_weights.detach().cpu().numpy()
    seq_len = attention_matrix.shape[0]  # 获取序列长度

    plt.figure(figsize=(8, 6))
    sns.heatmap(attention_matrix, cmap="viridis", annot=False)

    # 设置坐标轴刻度
    tick_positions = np.arange(0, seq_len, 5)  # 生成0,5,10...的刻度位置
    plt.xticks(
        ticks=tick_positions + 0.5,  # 对齐刻度到单元格中心
        labels=tick_positions.astype(int),  # 显示整数标签
        fontsize=12
    )
    plt.yticks(
        ticks=tick_positions + 0.5,
        labels=tick_positions.astype(int),
        fontsize=12
    )

    # 设置轴标签和标题
    plt.title(name, fontsize=22, fontweight='bold')
    plt.xlabel("Key Sequence Tokens", fontsize=18)
    plt.ylabel("Query Sequence Tokens", fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=14)  # 主刻度

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

