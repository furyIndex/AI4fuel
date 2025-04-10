from torch.utils.data import Dataset
import numpy as np

class MolDataset(Dataset):
    def __init__(self, datas, labels):
        self.data = datas
        self.label = labels

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)
