import logging
import os
from pathlib import Path
import torch
from sklearn.model_selection import train_test_split
from torchvision import transforms
from mnist import MNISTDataset
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models import MLPBase, MLPBatchNorm, MLPDropout, MLPDropoutLayerNorm, MLPLayerNorm
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import click
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from datamaestro import prepare_dataset

if __name__ == "__main__":

    IMAGE_SIZE = 28 * 28
    BATCH_SIZE = 300
    N_EPOCHS = 1000
    LR = 1e-3
    SAVE=True
    TRAIN_RATIO = 0.05

    # Prepare data
    ds = prepare_dataset("com.lecun.mnist")
    train_img, train_labels = ds.train.images.data(), ds.train.labels.data()
    n_samples = train_img.shape[0]
    test_size = n_samples - int(TRAIN_RATIO * n_samples)
    train_img, val_img, train_labels, val_labels = train_test_split(train_img, train_labels, test_size=test_size,
                                                                    random_state=42)
    test_img, test_labels = ds.test.images.data(), ds.test.labels.data()

    mean = train_img.mean() / 255.
    std = train_img.std() / 255.

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean, std)
                                    ])

    mnist_train = MNISTDataset(train_img, train_labels, transforms=transform)
    mnist_val = MNISTDataset(val_img, val_labels, transforms=transform)
    mnist_test = MNISTDataset(test_img, test_labels, transforms=transform)

    train_loader = DataLoader(mnist_train, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(mnist_val, batch_size=BATCH_SIZE, shuffle=False)

    criterion = nn.CrossEntropyLoss()

    for MLPClass in ["L2", "all"]: # MLPBase, MLPBatchNorm, MLPDropout, MLPLayerNorm, MLPDropoutLayerNorm,
        if MLPClass == "L2":
            print("L2")
            model = MLPBase().to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=0.01)
        elif MLPClass == "all":
            model = MLPDropoutLayerNorm().to(device)
            print(MLPDropoutLayerNorm.__name__, "L2")
            optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=0.01)
        else:
            print(MLPClass.__name__)
            model = MLPClass().to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=LR)

        loss_trace_train = []
        loss_trace_test = []
        acc_trace_train = []
        acc_trace_test = []
        for epoch in range(N_EPOCHS):
            epoch_loss_train = 0
            epoch_acc_train = 0
            model.train()
            for x, y in tqdm(train_loader):
                x, y = x.to(device), y.to(device)
                output = model(x)
                loss = criterion(output, y.view(-1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss_train += loss.item() * x.shape[0]
                epoch_acc_train += torch.eq(torch.argmax(output, dim=-1), y.view(-1)).sum().item()

            epoch_loss_train /= len(train_loader.dataset)
            epoch_acc_train /= len(train_loader.dataset)
            loss_trace_train.append(epoch_loss_train)
            acc_trace_train.append(epoch_acc_train)

            # validation
            if epoch % 25 == 0 or epoch == N_EPOCHS - 1:
                model.eval()
                epoch_loss_test = 0
                epoch_acc_test = 0
                with torch.no_grad():
                    for x, y in tqdm(val_loader):
                        x, y = x.to(device), y.to(device)
                        output = model(x)
                        loss = criterion(output, y.view(-1))
                        epoch_loss_test += loss.item() * x.shape[0]
                        epoch_acc_test += torch.eq(torch.argmax(output, dim=-1), y.view(-1)).sum().item()

                    epoch_loss_test /= len(val_loader.dataset)
                    epoch_acc_test /= len(val_loader.dataset)
                    loss_trace_test.append(epoch_loss_test)
                    acc_trace_test.append(epoch_acc_test)

                print(f"Epoch {epoch}, Loss Train : {epoch_loss_train}, Loss Val : {epoch_loss_test}")
                print(f"Epoch {epoch}, Acc Train : {epoch_acc_train}, Acc Val : {epoch_acc_test}")
            else:
                print(f"Epoch {epoch}, Loss Train : {epoch_loss_train}")
                print(f"Epoch {epoch}, Acc Train : {epoch_acc_train}")

            # checkpoint
            if SAVE and (epoch % 20 == 0 or epoch == N_EPOCHS - 1):
                if MLPClass == "L2" or MLPClass == "all":
                    model_name = f"models/{model.__class__.__name__}_L2.pth"
                else:
                    model_name = f"models/{model.__class__.__name__}.pth"
                torch.save({'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss_trace_train': loss_trace_train,
                            'loss_trace_test': loss_trace_test,
                            'acc_trace_train': acc_trace_train,
                            'acc_trace_test': acc_trace_test,
                            },
                           model_name)