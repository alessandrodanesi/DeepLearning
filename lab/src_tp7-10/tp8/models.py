import torch
import torch.nn as nn
import torch.nn.functional as F

# Vous utiliserez un réseau composé de 3 couches linéaires avec 100 sorties, suivis d'une
# couche linéaire pour la classification (10 classes  les chiffres de 0 à 9). Vous utiliserez un
# coût cross-entropique, des batchs de taille 300, et 1000 itérations (epochs).
# Question
#


class MLPBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 100)
        self.fc4 = nn.Linear(100, 10)

    def forward(self, x):
        bsize = x.size(0)
        x = x.view(bsize, -1)
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        out = self.fc4(out)
        return out


class MLPDropout(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 100)
        self.dropout1 = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(100, 100)
        self.dropout2 = nn.Dropout(p=0.2)
        self.fc3 = nn.Linear(100, 100)
        self.dropout3 = nn.Dropout(p=0.2)
        self.fc4 = nn.Linear(100, 10)

    def forward(self, x):
        bsize = x.size(0)
        x = x.view(bsize, -1)
        out = F.relu(self.fc1(x))
        out = self.dropout1(out)
        out = F.relu(self.fc2(out))
        out = self.dropout2(out)
        out = F.relu(self.fc3(out))
        out = self.dropout3(out)
        out = self.fc4(out)
        return out


class MLPBatchNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 100)
        self.norm1 = nn.BatchNorm1d(100)
        self.fc2 = nn.Linear(100, 100)
        self.norm2 = nn.BatchNorm1d(100)
        self.fc3 = nn.Linear(100, 100)
        self.norm3 = nn.BatchNorm1d(100)
        self.fc4 = nn.Linear(100, 10)

    def forward(self, x):
        bsize = x.size(0)
        x = x.view(bsize, -1)
        out = F.relu(self.norm1(self.fc1(x)))
        out = F.relu(self.norm2(self.fc2(out)))
        out = F.relu(self.norm3(self.fc3(out)))
        out = self.fc4(out)
        return out


class MLPLayerNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 100)
        self.norm1 = nn.LayerNorm(100)
        self.fc2 = nn.Linear(100, 100)
        self.norm2 = nn.LayerNorm(100)
        self.fc3 = nn.Linear(100, 100)
        self.norm3 = nn.LayerNorm(100)
        self.fc4 = nn.Linear(100, 10)

    def forward(self, x):
        bsize = x.size(0)
        x = x.view(bsize, -1)
        out = F.relu(self.norm1(self.fc1(x)))
        out = F.relu(self.norm2(self.fc2(out)))
        out = F.relu(self.norm3(self.fc3(out)))
        out = self.fc4(out)
        return out


class MLPDropoutLayerNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 100)
        self.norm1 = nn.LayerNorm(100)
        self.dropout1 = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(100, 100)
        self.norm2 = nn.LayerNorm(100)
        self.dropout2 = nn.Dropout(p=0.2)
        self.fc3 = nn.Linear(100, 100)
        self.norm3 = nn.LayerNorm(100)
        self.dropout3 = nn.Dropout(p=0.2)
        self.fc4 = nn.Linear(100, 10)

    def forward(self, x):
        bsize = x.size(0)
        x = x.view(bsize, -1)
        out = F.relu(self.norm1(self.fc1(x)))
        out = self.dropout1(out)
        out = F.relu(self.norm2(self.fc2(out)))
        out = self.dropout2(out)
        out = F.relu(self.norm3(self.fc3(out)))
        out = self.dropout3(out)
        out = self.fc4(out)
        return out