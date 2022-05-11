#  TODO:  Question 3 : Prédiction de séries temporelles
from temperatures_utils import read_temps, collate_batch, TempDataset
from rnn import RNN, MonoVariateRNN
from generate import forecast
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import os
from matplotlib import pyplot as plt
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def teacher_forcing(model, input_seq, target_seq, loss_func, k=1):
    x_target_seq = target_seq[:-k]
    y_target_seq = target_seq[k:]
    _, h_init = model(input_seq)
    h_last_l, h = model(x_target_seq, h_init=h_init)
    y_target_seq_hat = model.decode(h_last_l)
    lloss = loss_func(y_target_seq.flatten(end_dim=1), y_target_seq_hat.flatten(end_dim=1))
    return lloss


def train_epoch(model1, model2, loss_func, optimizer1, optimizer2, loader, epoch_num):
    epoch_loss_train1 = 0
    epoch_loss_train2 = 0

    for n_batch, batch in enumerate(loader):
        t = (n_batch + epoch_num * (N_train // BATCH_SIZE)) * BATCH_SIZE
        input_seq, _ = batch
        input_seq = input_seq.to(device)
        x = input_seq[:-n_steps]
        y = input_seq[-n_steps:]
        loss_teacher1, loss_forecast1 = 0, 0
        loss_teacher2, loss_forecast2 = 0, 0
        # Teacher forcing :
        if TEACHER:
            loss_teacher1 = teacher_forcing(model1, x, y, loss_func, k=1)
            for i in range(model2.n_models):
                loss_teacher2 += teacher_forcing(model2.models[i], x[:, :, i].unsqueeze(-1), y[:, :, i].unsqueeze(-1),
                                                 loss_func, k=1)

        # Forecast :
        if FORECAST:
            preds1 = forecast(model1, x, n_steps)
            loss_forecast1 = loss_func(preds1.flatten(end_dim=1), y.flatten(end_dim=1))
            for i in range(model2.n_models):
                preds2 = forecast(model2.models[i], x[:, :, i].unsqueeze(-1), n_steps)
                loss_forecast2 += loss_func(preds2.flatten(end_dim=1), y[:, :, i].unsqueeze(-1).flatten(end_dim=1))

        train_loss1 = TEACHER * loss_teacher1 + FORECAST * loss_forecast1
        train_loss2 = (TEACHER * loss_teacher2 + FORECAST * loss_forecast2) / model2.n_models
        train_loss1.backward()
        train_loss2.backward()

        optimizer1.step()
        optimizer1.zero_grad()

        optimizer2.step()
        optimizer2.zero_grad()

        writer.add_scalars("Loss / Train", {"Multivariate": train_loss1.detach(),
                                            "Monovariate": train_loss2.detach()}, t)

        epoch_loss_train1 += train_loss1.detach() * input_seq.shape[1]
        epoch_loss_train2 += train_loss2.detach() * input_seq.shape[1]

    epoch_loss_train1 /= len(loader.dataset)
    epoch_loss_train2 /= len(loader.dataset)
    writer.add_scalars("Loss Epoch / Train", {"Multivariate": epoch_loss_train1.detach(),
                                              "Monovariate": epoch_loss_train2.detach()}, epoch_num)

    print(f"Epoch {epoch_num} Train Loss Multivariate : {epoch_loss_train1:.4f}")
    print(f"Epoch {epoch_num} Train Loss Monovariate : {epoch_loss_train2:.4f}")


def val_epoch(model1, model2, loss_func, loader, epoch_num):
    epoch_loss_val1 = 0
    epoch_loss_val2 = 0

    for n_batch, batch in enumerate(loader):
        t = (n_batch + epoch_num * (N_train // BATCH_SIZE)) * BATCH_SIZE
        input_seq, _ = batch
        input_seq = input_seq.to(device)
        x = input_seq[:-n_steps]

        loss1, loss2 = 0, 0
        # forecast :
        y = input_seq[-n_steps:]
        preds1 = forecast(model1, x, n_steps)
        loss1 = loss_func(preds1.flatten(end_dim=1), y.flatten(end_dim=1))
        for i in range(model2.n_models):
            preds2 = forecast(model2.models[i], x[:, :, i].unsqueeze(-1), n_steps)
            loss2 += loss_func(preds2.flatten(end_dim=1), y[:, :, i].unsqueeze(-1).flatten(end_dim=1))

        epoch_loss_val1 += loss1.detach() * input_seq.shape[1]
        epoch_loss_val2 += loss2.detach() * input_seq.shape[1]
    epoch_loss_val1 /= len(loader.dataset)
    epoch_loss_val2 /= len(loader.dataset)
    writer.add_scalars("Loss Epoch / Test", {"Multivariate": epoch_loss_val1.detach(),
                                             "Monovariate": epoch_loss_val2.detach()}, epoch_num)
    print(f"Epoch {epoch_num} Test Loss Multivariate: {epoch_loss_val1:.4f}")
    print(f"Epoch {epoch_num} Test Loss Monovariate: {epoch_loss_val2:.4f}")


if __name__ == "__main__":
    path_data = "data/"
    path_temps_train = os.path.join(path_data, "tempAMAL_train.csv")
    path_temps_test = os.path.join(path_data, "tempAMAL_test.csv")

    data_train = read_temps(path_temps_train)
    data_test = read_temps(path_temps_test)
    cities = pd.read_csv(path_temps_train).columns.tolist()[1:]

    maxtemp = data_train.max()
    mintemp = data_train.min()

    N_CITIES = 10
    INPUT_DIM = N_CITIES
    OUTPUT_DIM = N_CITIES
    BATCH_SIZE = 128
    N_EPOCHS = 10
    LR = 4e-3
    LATENT_DIM_MULTI = 100
    LATENT_DIM_MONO = 10
    N_LAYERS = 1
    MAX_LEN = 7 * 24
    TEACHER = True
    FORECAST = True

    city_ids = np.arange(len(cities))
    np.random.shuffle(city_ids)
    sub_cities = city_ids[:N_CITIES].tolist()
    # sub_cities = [6, 18, 3, 28]  # 6 Las Vegas, 28 Montreal, 3 Seattle, 18 Indianapolis
    print([cities[i] for i in sub_cities])
    temps_train = TempDataset(data_train[:, sub_cities], MAX_LEN, maxtemp, mintemp)
    temps_test = TempDataset(data_test[:, sub_cities], MAX_LEN, maxtemp, mintemp)
    N_train = len(temps_train)
    N_test = len(temps_test)

    train_loader = DataLoader(temps_train, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
    test_loader = DataLoader(temps_test, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)

    multi_rnn = RNN(INPUT_DIM, LATENT_DIM_MULTI, OUTPUT_DIM, N_LAYERS,
                    activation_enc=nn.Tanh(), activation_dec=nn.Tanh(), device=device)

    mono_rnn = MonoVariateRNN(INPUT_DIM, LATENT_DIM_MONO, N_LAYERS, activation_enc=nn.Tanh(), activation_dec=nn.Tanh(),
                              device=device)

    criterion = nn.MSELoss()
    multi_optimizer = torch.optim.Adam(multi_rnn.parameters(), lr=LR)
    mono_optimizer = torch.optim.Adam(mono_rnn.parameters(), lr=LR)

    writer = SummaryWriter(comment="")

    n_steps = 24
    try:
        for epoch in range(N_EPOCHS):
            train_epoch(multi_rnn, mono_rnn, criterion, multi_optimizer, mono_optimizer, train_loader, epoch)
            val_epoch(multi_rnn, mono_rnn, criterion, test_loader, epoch)

        writer.flush()
        writer.close()
    except KeyboardInterrupt:
        writer.flush()
        writer.close()

    # Visual check
    cities_ = [cities[i] for i in sub_cities]
    seq, labels = next(iter(test_loader))
    seq = seq.to(device)
    y_seq = seq[-n_steps:]
    x_seq = seq[:-n_steps]
    with torch.no_grad():
        h_last_layer, h_last = multi_rnn(x_seq)
        x_hat_seq = multi_rnn.decode(h_last_layer)
        y_preds = forecast(multi_rnn, x_seq, n_steps)

    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    for i, ax in enumerate(axes.flatten()):
        if i >= len(cities_):
            break
        ax.plot(seq[:, 2, i].cpu(), label="Truth")
        ax.plot(np.arange(x_seq.shape[0], x_seq.shape[0] + n_steps), y_preds[:, 2, i].detach().cpu(), label="Forecast")
        ax.plot(np.arange(x_seq.shape[0]), x_hat_seq[:, 2, i].detach().cpu(), label="Decoded Emb")
        ax.set_title(f"Label {cities_[int(labels[2, i].cpu().item())]}", fontsize=7)
    plt.show()

    with torch.no_grad():
        x_hat_seq = torch.zeros_like(x_seq)
        y_preds = torch.zeros_like(y_seq)
        for i in range(mono_rnn.n_models):
            h_last_layer, h_last = mono_rnn.models[i](x_seq[:, :, i].unsqueeze(-1))
            x_hat_seq[:, :, i] = mono_rnn.models[i].decode(h_last_layer)
            y_preds[:, :, i] = forecast(mono_rnn.models[i], x_seq[:, :, i].unsqueeze(-1), n_steps)

    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    for i, ax in enumerate(axes.flatten()):
        if i >= len(cities_):
            break
        ax.plot(seq[:, 2, i].cpu(), label="Truth")
        ax.plot(np.arange(x_seq.shape[0], x_seq.shape[0] + n_steps), y_preds[:, 2, i].detach().cpu(), label="Forecast")
        ax.plot(np.arange(x_seq.shape[0]), x_hat_seq[:, 2, i].detach().cpu(), label="Decoded Emb")
        ax.set_title(f"Label {cities_[int(labels[2, i].cpu().item())]}", fontsize=7)
    plt.show()