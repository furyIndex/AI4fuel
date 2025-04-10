import torch
import torch.nn as nn
from scipy.stats import spearmanr
import torch.nn.functional as F



class EmbeddingMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, output_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)    
        )

    def forward(self, x):
        return self.net(x)





# class ContrastiveLoss(nn.Module):
#     def __init__(self, margin=2.0):
#         super().__init__()
#         self.margin = margin
#
#     def forward(self, embeddings, labels):
#         # 计算样本对的标签差异
#         y_diff = torch.abs(labels.unsqueeze(0) - labels.unsqueeze(1))  # [B,B]
#
#         # 生成相似性掩码（相似=1，不相似=0）
#         similarity_mask = (y_diff < 0.1*torch.std(labels)).float()  # 阈值自适应
#
#
#         # embeddings = F.normalize(embeddings, p=2, dim=1)  # L2归一化
#         # 计算嵌入的欧氏距离
#         distances = torch.cdist(embeddings, embeddings, p=2)
#
#         # 对比损失计算
#         loss = (similarity_mask * distances.pow(2) +
#                 (1 - similarity_mask) * torch.relu(self.margin - distances).pow(2))
#         return loss.mean()






class ContrastiveLoss(nn.Module):
    def __init__(self, margin=2.0):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings, labels):
        # 计算样本对的标签差异
        y_diff = torch.abs(labels.unsqueeze(0) - labels.unsqueeze(1))  # [B,B]
        std_label = torch.std(labels)
        similarity_mask = (y_diff < 0.1 * std_label).float()  # 自适应阈值

        # 计算嵌入的欧氏距离
        distances = torch.cdist(embeddings, embeddings, p=2)


        # 生成负样本掩码：非正样本且距离在margin内
        negative_mask = (similarity_mask == 0) & (self.margin - distances > 0)
        negative_mask = negative_mask.bool()  # 转换为布尔类型

        batch_size = embeddings.size(0)
        mask = torch.zeros_like(similarity_mask)

        # 对每个样本，随机选择一个负样本
        for i in range(batch_size):
            valid_negatives = torch.where(negative_mask[i])[0]
            if valid_negatives.numel() > 0:  # 存在有效负样本
                selected_idx = torch.randint(0, len(valid_negatives), (1,))
                selected_j = valid_negatives[selected_idx]
                mask[i, selected_j] = 1.0

        # 计算对比损失
        positive_loss = similarity_mask * distances.pow(2)
        negative_loss = mask * torch.relu(self.margin - distances).pow(2)
        loss = (positive_loss + negative_loss).mean()

        return loss














def embedding_val_func(embeddings, labels):
    # 计算Spearman相关系数
    embed_dist = torch.cdist(embeddings, embeddings).flatten().cpu().numpy()
    label_diff = torch.abs(labels.unsqueeze(0)-labels.unsqueeze(1)).flatten().cpu().numpy()
    res = spearmanr(embed_dist, label_diff)

    return res