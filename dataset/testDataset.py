from torch.utils.data import Dataset
from prepare import prepare_data
import numpy as np
import pandas as pd

class testDataset(Dataset):
    def __init__(self, path):
        self.data = pd.read_csv(path, header=1)
        self.labels = np.asarray(self.data.iloc[:, -1])

    def __getitem__(self, index):
        single_label = self.labels[index]
        single_data = np.asarray(self.data.iloc[index][:-1]).astype(np.float32)
        return single_data, single_label

    def __len__(self):
        return len(self.labels)


if __name__ == '__main__':
    dataset = testDataset()
    print(dataset.__len__())
    print(dataset.__getitem__(1))

