
import torch
import torch.nn as nn
from scipy.stats import spearmanr



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












class ContrastiveLoss(nn.Module):
    def __init__(self, margin=2.0):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings, labels):

        y_diff = torch.abs(labels.unsqueeze(0) - labels.unsqueeze(1))
        std_label = torch.std(labels)
        similarity_mask = (y_diff < 0.1 * std_label).float()

        distances = torch.cdist(embeddings, embeddings, p=2)

        negative_mask = (similarity_mask == 0) & (self.margin - distances > 0)
        negative_mask = negative_mask.bool()
        batch_size = embeddings.size(0)
        mask = torch.zeros_like(similarity_mask)

        for i in range(batch_size):
            valid_negatives = torch.where(negative_mask[i])[0]
            if valid_negatives.numel() > 0:
                selected_idx = torch.randint(0, len(valid_negatives), (1,))
                selected_j = valid_negatives[selected_idx]
                mask[i, selected_j] = 1.0

        positive_loss = similarity_mask * distances.pow(2)
        negative_loss = mask * torch.relu(self.margin - distances).pow(2)

        n_pos = similarity_mask.sum() + 1e-8
        n_neg = mask.sum() + 1e-8
        positive_loss = positive_loss.sum() / n_pos
        negative_loss = negative_loss.sum() / n_neg
        loss = positive_loss + negative_loss

        return loss














def embedding_val_func(embeddings, labels):

    embed_dist = torch.cdist(embeddings, embeddings).flatten().cpu().numpy()
    label_diff = torch.abs(labels.unsqueeze(0)-labels.unsqueeze(1)).flatten().cpu().numpy()
    res = spearmanr(embed_dist, label_diff)

    return res