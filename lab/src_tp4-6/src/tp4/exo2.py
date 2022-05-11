#  TODO:  Question 2 : prédiction de la ville correspondant à une séquence
from temperatures_utils import read_temps, collate_batch, TempDataset
from rnn import RNN
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import os
from matplotlib import pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":

    # Prepare Data
    path_data = "data/"
    path_temps_train = os.path.join(path_data, "tempAMAL_train.csv")
    path_temps_test = os.path.join(path_data, "tempAMAL_test.csv")

    data_train = read_temps(path_temps_train)
    data_test = read_temps(path_temps_test)
    cities = pd.read_csv(path_temps_train).columns.tolist()[1:]

    maxtemp = data_train.max()
    mintemp = data_train.min()

    N_CITIES = 10
    INPUT_DIM = 1
    OUTPUT_DIM = N_CITIES
    BATCH_SIZE = 128
    N_EPOCHS = 20
    LR = 1e-3
    LATENT_DIM = 20
    N_LAYERS = 2
    MAX_LEN = 7 * 24

    city_ids = np.arange(len(cities))
    np.random.shuffle(city_ids)
    sub_cities = city_ids[:N_CITIES].tolist()
    # sub_cities = [3, 6, 28]  # 3 Seattle, 6 Las Vegas, 18 Indianapolis, 28 Montreal
    print(sub_cities)
    print([cities[i] for i in sub_cities])
    temps_train = TempDataset(data_train[:, sub_cities], MAX_LEN, maxtemp, mintemp, classif=True)
    temps_test = TempDataset(data_test[:, sub_cities], MAX_LEN, maxtemp, mintemp, classif=True)
    N_train = len(temps_train)
    N_test = len(temps_test)

    train_loader = DataLoader(temps_train, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
    test_loader = DataLoader(temps_test, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)

    rnn = RNN(INPUT_DIM, LATENT_DIM, OUTPUT_DIM, N_LAYERS, activation_enc=nn.ReLU(),
              activation_dec=nn.Identity(), device=device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)

    writer = SummaryWriter(comment="")

    try:
        for epoch in range(N_EPOCHS):
            loss_train = 0
            acc_train = 0
            # Epoch Train
            for n_batch, batch in enumerate(train_loader):
                t = (n_batch + epoch * (N_train // BATCH_SIZE)) * BATCH_SIZE
                seq_batches, labels = batch
                labels = labels.to(device)
                seq_batches = seq_batches.to(device)
                h_last_layer, h_last = rnn(seq_batches)
                outputs = rnn.decoder(h_last[-1])
                preds = outputs.argmax(dim=-1)
                loss = criterion(outputs, labels.long())
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                writer.add_scalar("Loss Train", loss.detach(), t)
                loss_train += loss.detach() * seq_batches.shape[1]
                acc_train += torch.eq(preds, labels).sum().item()

            acc_train /= len(train_loader.dataset)
            loss_train /= len(train_loader.dataset)

            loss_test = 0
            acc_test = 0
            # Epoch Train
            for n_batch, batch in enumerate(test_loader):
                t = (n_batch + epoch * (N_test // BATCH_SIZE)) * BATCH_SIZE
                seq_batches, labels = batch
                labels = labels.to(device)
                seq_batches = seq_batches.to(device)

                with torch .no_grad():
                    h_last_layer, h_last = rnn(seq_batches)
                    outputs = rnn.decoder(h_last[-1])
                    preds = outputs.argmax(dim=-1)
                    loss = criterion(outputs, labels.long())

                writer.add_scalar("Loss Val", loss.detach(), t)
                loss_test += loss.detach() * seq_batches.shape[1]
                acc_test += torch.eq(preds, labels).sum().item()

            acc_test /= len(test_loader.dataset)
            loss_test /= len(test_loader.dataset)

            writer.add_scalar("Loss Epoch Train", loss_train, epoch)
            writer.add_scalar("Acc Epoch Train", acc_train, epoch)
            writer.add_scalar("(0-1) Loss Epoch Train", 1-acc_train, epoch)
            writer.add_scalar("Loss Epoch Val", loss_test, epoch)
            writer.add_scalar("Acc Epoch Val", acc_test, epoch)
            writer.add_scalar("(0-1) Loss Epoch Val", 1-acc_test, epoch)

            print(f"Epoch {epoch} Train Loss : {loss_train:.4f}, Val Loss : {loss_test:.4f}")
            print(f"Epoch {epoch} Train Acc : {acc_train*100:.1f}%, Val Acc : {acc_test*100:.1f}%")
        writer.flush()
        writer.close()
    except KeyboardInterrupt:
        writer.flush()
        writer.close()

    # Visual check
    cities_ = [cities[i] for i in sub_cities]
    seq, labels = next(iter(test_loader))
    with torch.no_grad():
        h_last_layer, h_last = rnn.forward(seq)
        outputs = rnn.decoder(h_last[-1])
        preds = outputs.argmax(dim=1)

    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    for i, ax in enumerate(axes.flatten()):
        ax.plot(seq[:, i, :].cpu(),)
        ax.set_title(f"Prediction {cities_[int(preds[i].cpu().item())]}\n "
                     f"Label {cities_[int(labels[i].cpu().item())]}",
                     fontsize=7)
        ax.xaxis.set_visible(False)
    plt.show()