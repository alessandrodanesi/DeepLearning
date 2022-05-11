import torch
from torch import nn
from torch.utils.data import DataLoader
from rnn import RNN
from generate import generate
from embedding import OneHotEmbedding
from text_utils import TextDataset, VOCAB_SIZE, collate_fn, code2string, eos
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import string
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    path_speech = "./data/trump_full_speech2.txt"
    with open(path_speech, "r") as f:
        trump_speech = f.read()

    INPUT_DIM = VOCAB_SIZE
    OUTPUT_DIM = VOCAB_SIZE
    BATCH_SIZE = 1000
    N_EPOCHS = 500
    LR = 1e-3
    LATENT_DIM = 200
    N_LAYERS = 2
    MAX_LEN = 100
    MIN_LEN = 30

    N = int(len(trump_speech) * 0.9)
    speech_train = trump_speech[:N]
    speech_test = trump_speech[N:]
    ds_train = TextDataset(speech_train, maxlen=MAX_LEN, minlen=MIN_LEN)
    ds_test = TextDataset(speech_test, maxlen=MAX_LEN, minlen=MIN_LEN)
    loader_train = DataLoader(ds_train, collate_fn=collate_fn, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    loader_test = DataLoader(ds_test, collate_fn=collate_fn, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

    onehot = OneHotEmbedding(VOCAB_SIZE, device=device)
    rnn = RNN(INPUT_DIM, LATENT_DIM, OUTPUT_DIM, N_LAYERS, activation_enc=nn.Tanh(),
              activation_dec=nn.Identity(), device=device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)

    # checkpoint = torch.load("rnn_trump.pth")
    # rnn.load_state_dict(checkpoint['model_state_dict'])
    writer = SummaryWriter()
    loss_trace_train = []
    loss_trace_test = []
    for epoch in range(N_EPOCHS):
        epoch_loss = 0
        for n, data in enumerate(tqdm(loader_train)):
            timestep = (n + epoch * (len(loader_train.dataset) // BATCH_SIZE)) * BATCH_SIZE
            data = data.to(device)

            seq_len = data.shape[0]
            x_seq = data[:-1]
            y_seq = data[1:]
            h_last_layer, h_last = rnn(onehot(x_seq))
            y_preds = rnn.decode(h_last_layer)
            loss = criterion(y_preds.flatten(end_dim=1), y_seq.flatten(end_dim=1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * data.shape[1]
            # print(f"Loss {loss}")
            # writer.add_scalar("Loss", loss.item(), timestep)
        epoch_loss /= len(loader_train.dataset)
        loss_trace_train.append(epoch_loss)

        epoch_loss_test = 0
        for n, data in enumerate(tqdm(loader_test)):
            timestep = (n + epoch * (len(loader_test.dataset) // BATCH_SIZE)) * BATCH_SIZE
            data = data.to(device)
            seq_len = data.shape[0]
            x_seq = data[:-1]
            y_seq = data[1:]
            with torch.no_grad():
                h_last_layer, h_last = rnn(onehot(x_seq))
                y_preds = rnn.decode(h_last_layer)
                loss = criterion(y_preds.flatten(end_dim=1), y_seq.flatten(end_dim=1))
                epoch_loss_test += loss.item() * data.shape[1]
        epoch_loss_test /= len(loader_test.dataset)
        loss_trace_test.append(epoch_loss_test)
        print(f"Epoch {epoch} Loss Train: {epoch_loss}, Loss Test: {epoch_loss_test}")

        if epoch % 50 == 0:
            with torch.no_grad():
                # generate
                data = data[:, 0]
                T = data.shape[0] - 1
                x_seq = x_seq[:, 0]
                y_preds = y_preds[:, 0].argmax(dim=-1)
                y_seq = y_seq[:, 0]
                print("Test Prediction")
                print(f"Input : ||{code2string(x_seq)}||")
                print(f"Predictions : ||{code2string(y_preds)}||")
                print(f"True : ||{code2string(y_seq)}||")
                print("")
                print("Test Generation last 2")
                preds_chars = generate(rnn, onehot, eos, code2string(data[:-2]), maxlen=T)
                print(f"Input : ||{code2string(data[:-2])}||")
                print(f"Predictions : ||{preds_chars}||")
                print(f"True : ||{code2string(data)}||")
                print("")
                print("Test Generation with input")
                preds_chars = generate(rnn, onehot, eos, code2string(data[:10]), maxlen=T+30,)
                print(f"Input : ||{code2string(data[:10])}||")
                print(f"Predictions : ||{preds_chars}||")
                print(f"True : ||{code2string(data)}||")

                print("Test Generation without input")
                start = string.ascii_letters[-26:][np.random.randint(0, 25)]
                preds_chars = generate(rnn, onehot, eos, start, maxlen=100)
                print(f"Predictions Greddy: ||{preds_chars}||")
                preds_chars = generate(rnn, onehot, eos, start, maxlen=100, stochastic=True)
                print(f"Predictions Stochastique: ||{preds_chars}||")
        if (epoch % 50 == 0) and (epoch > 0):
            torch.save({'epoch': epoch,
                        'model_state_dict': rnn.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss_train': epoch_loss,
                        'loss_trace_train': loss_trace_train,
                        'loss_test': epoch_loss_test,
                        'loss_trace_test': loss_trace_test,
                        "embedding": onehot},
                       f"rnn_trump_onehot_latent{LATENT_DIM}_nlayers{N_LAYERS}.pth")

    preds_chars = generate(rnn, onehot, eos, "Trump is a ", maxlen=150)
    print(preds_chars)
    writer.close()
