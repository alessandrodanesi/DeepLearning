import torch
from torch.utils.data import Dataset


class MNISTDataset(Dataset):
    def __init__(self, data, labels, transforms=None):
        super().__init__()
        self.data = data
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        if idx >= self.__len__() or idx < 0:
            raise IndexError

        # load label
        label = torch.LongTensor([self.labels[idx]])
        # load image
        img = self.data[idx]
        return self.transforms(img), label



