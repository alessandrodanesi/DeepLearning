import numpy as np
import string
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from textloader import pad_ix, pad, eos_ix, eos, code2string, id2lettre, TextDataset, collate_fn
from generate import generate, generate_beam, generate_nucleus
from rnn import RNN, LSTM, GRU
from itertools import chain
import logging
from tqdm import tqdm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.basicConfig(level=logging.INFO)


def maskedCrossEntropy(output, target, padcode):
    crossentropy = nn.CrossEntropyLoss(reduction="none")
    output = output.flatten(end_dim=1)
    target = target.flatten(end_dim=1)
    mask = target != padcode
    loss_ = crossentropy(output, target.long())
    loss_ *= mask
    return loss_.mean()


if __name__ == "__main__":
    path_speech = "./data/trump_full_speech2.txt"
    with open(path_speech, "r") as f:
        trump_speech = f.read()

    VOCAB_SIZE = len(id2lettre)  # includes padding char
    INPUT_DIM = VOCAB_SIZE
    OUTPUT_DIM = VOCAB_SIZE
    BATCH_SIZE = 500
    N_EPOCHS = 200
    LR = 1e-3
    LATENT_DIM = 200
    EMBED_DIM = 60
    N_LAYERS = 2
    MAX_LEN = 100
    MIN_LEN = 80
    SAVE = True

    N = int(len(trump_speech) * 0.9)
    speech_train = trump_speech[:N]
    speech_test = trump_speech[N:]
    ds_train = TextDataset(speech_train, maxlen=MAX_LEN, minlen=MIN_LEN)
    ds_test = TextDataset(speech_test, maxlen=MAX_LEN, minlen=MIN_LEN)
    loader_train = DataLoader(ds_train, collate_fn=collate_fn, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
    loader_test = DataLoader(ds_test, collate_fn=collate_fn, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

    # model = RNN(EMBED_DIM, LATENT_DIM, OUTPUT_DIM, N_LAYERS, activation_enc=nn.Tanh(),
    #             activation_dec=nn.Identity(), device=device)

    model = LSTM(EMBED_DIM, LATENT_DIM, OUTPUT_DIM, N_LAYERS, device)
    # model = GRU(EMBED_DIM, LATENT_DIM, OUTPUT_DIM, N_LAYERS, device)
    embedding = nn.Embedding(VOCAB_SIZE, EMBED_DIM, padding_idx=pad_ix).to(device)
    params = [model.parameters(), embedding.parameters()]
    print(model.name)
    optimizer = torch.optim.Adam(chain(*params), lr=LR)

    # writer = SummaryWriter()
    loss_trace_train = []
    loss_trace_test = []
    for epoch in range(N_EPOCHS):
        epoch_loss_train = 0
        for n, data in enumerate(tqdm(loader_train)):
            timestep = (n + epoch * (len(loader_train.dataset) // BATCH_SIZE)) * BATCH_SIZE
            data = data.to(device)
            seq_len = data.shape[0]
            x_seq = data[:-1]
            y_seq = data[1:]

            h_last_layer, h_last = model(embedding(x_seq.long()))
            y_preds = model.decode(h_last_layer)

            loss = maskedCrossEntropy(y_preds, y_seq, pad_ix)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss_train += loss.item() * data.shape[1]
        epoch_loss_train /= len(loader_train.dataset)
        loss_trace_train.append(epoch_loss_train)

        epoch_loss_test = 0
        for n, data in enumerate(tqdm(loader_test)):
            timestep = (n + epoch * (len(loader_test.dataset) // BATCH_SIZE)) * BATCH_SIZE
            data = data.to(device)
            seq_len = data.shape[0]
            x_seq = data[:-1]
            y_seq = data[1:]
            with torch.no_grad():
                h_last_layer, h_last = model(embedding(x_seq.long()))
                y_preds = model.decode(h_last_layer)
                loss = maskedCrossEntropy(y_preds, y_seq, pad_ix)
                epoch_loss_test += loss.item() * data.shape[1]
        epoch_loss_test /= len(loader_test.dataset)
        loss_trace_test.append(epoch_loss_test)

        print(f"Epoch {epoch}, Loss Train : {epoch_loss_train}, Loss Test : {epoch_loss_test}")
        if (epoch % 100 == 0) and (epoch > 0):
            with torch.no_grad():
                # generate
                data = data[:, 0]
                T = torch.where(data.eq(eos_ix))[0][0].item()
                data = data[:T+1]
                x_seq = x_seq[:T, 0]
                y_preds = y_preds[:T, 0].argmax(dim=-1)
                y_seq = y_seq[:T, 0]
                print("*** Test Prediction ***")
                print(f"Input : {code2string(x_seq)}")
                print(f"Pred  : {code2string(y_preds)}")
                print(f"True  : {code2string(y_seq)}")
                print("")
                print("*** Test Generation last 2 ***")
                preds_chars = generate(model, embedding, model.decoder, eos, code2string(data[:-2]), maxlen=T+1)
                print(f"Input : {code2string(data[:-2])}")
                print(f"Pred  : {preds_chars}")
                print(f"True  : {code2string(data)}")
                print("")
                print("*** Test Generation with input (Greedy) ***")
                preds_chars = generate(model, embedding, model.decoder, eos, code2string(data[:10]), maxlen=T + 30, )
                print(f"Input : {code2string(data[:10])}")
                print(f"Pred  : {preds_chars}")
                print(f"True  : {code2string(data)}")
                print("")
                print("*** Test Generation without input ***")
                start = string.ascii_uppercase[np.random.randint(0, 25)]
                preds_chars = generate(model, embedding, model.decoder, eos, start, maxlen=50)
                print(f"Greedy      : {preds_chars}")
                preds_chars = generate(model, embedding, model.decoder, eos, start, maxlen=50, stochastic=True)
                print(f"Sampling    : {preds_chars}")
                preds_chars = generate_beam(model, embedding, model.decoder, eos, 4, start, maxlen=50)
                print(f"Beam Search : {preds_chars}")
                preds_chars = generate_nucleus(model, embedding, model.decoder, eos, p=0.9, k=5, mode="pnucleus", start=start, maxlen=50)
                print(f"P_nucleus   : {preds_chars}")
                preds_chars = generate_nucleus(model, embedding, model.decoder, eos, p=0.9, k=5, mode="topk", start=start, maxlen=50)
                print(f"Top_K_Sampl : {preds_chars}")

        if SAVE and (epoch % 100 == 0 or epoch == N_EPOCHS - 1) and (epoch > 0):
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss_train': epoch_loss_train,
                        'loss_trace_train': loss_trace_train,
                        'loss_test': epoch_loss_test,
                        'loss_trace_test': loss_trace_test,
                        "embedding": embedding},
                       f"models/{model.name}_long_emb{EMBED_DIM}_latent{LATENT_DIM}_nlayers{N_LAYERS}.pth")