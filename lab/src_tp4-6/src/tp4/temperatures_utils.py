import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import logging
from torch.utils.data import Dataset
logging.basicConfig(level=logging.INFO)


def fill_na(mat):
    ix, iy = np.where(np.isnan(mat))
    for i, j in zip(ix, iy):
        if np.isnan(mat[i + 1, j]):
            mat[i, j] = mat[i - 1, j]
        else:
            mat[i, j] = (mat[i - 1, j] + mat[i + 1, j]) / 2.
    return mat


def read_temps(path):
    """Lit le fichier de températures"""
    temps = torch.tensor(fill_na(np.array(pd.read_csv(path).iloc[:, 1:])), dtype=torch.float)
    return temps


def collate_batch(batch):
    # Pick up  a subsequence at random
    seq_len = batch[0][0].shape[0]
    # T = np.random.choice(np.arange(seq_len // 2, seq_len))
    T = seq_len
    # dimension returned seq_len, batch_size, dim
    samples = torch.stack([seq[:T] for seq, label in batch]).permute(1, 0, 2)
    labels = torch.Tensor([label for seq, label in batch])

    return [samples, labels]


class TempDataset(Dataset):

    def __init__(self, data, seq_max_len, maxtemp, mintemp, classif=False):
        super().__init__()
        self.data = (2 * data - (maxtemp + mintemp)) / (maxtemp - mintemp)
        self.N = self.data.shape[0]
        self.N_labels = self.data.shape[1]
        self.seq_max_len = seq_max_len
        self.classif = classif

    def __len__(self):
        return (self.N - self.seq_max_len + 1) * 1  # self.N_labels

    def __getitem__(self, idx):

        t_0 = idx % (self.N - self.seq_max_len + 1)
        if self.classif:
            # label = idx // (self.N - self.seq_max_len + 1)
            # pick one label at random
            label = np.random.choice(np.arange(self.N_labels), size=1).tolist()
        else:
            label = np.arange(self.N_labels).tolist()
        return self.data[t_0:t_0 + self.seq_max_len, label], label


