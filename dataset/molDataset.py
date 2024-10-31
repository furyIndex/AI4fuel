from torch.utils.data import Dataset
from prepare import prepare_data
import numpy as np

class MolDataset(Dataset):
    def __init__(self, datas, labels):
        self.data = datas
        self.label = labels

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    data = []
    smiles_list = prepare_data.smilesFromExcel()
    for smiles in smiles_list:
        data.append(prepare_data.getMorganFromSmiles(smiles, 2048))
    data = np.array(data)
    labels = prepare_data.hoVFromExcel()
    dataset = MolDataset(data, labels)
    element = dataset[0]
    print(element)
