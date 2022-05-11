import logging
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import clip_grad_norm_
import torch.nn as nn
import torch
from tqdm import tqdm
from seq2seq import Encoder, Decoder
import numpy as np
from itertools import chain
from torch.utils.tensorboard import SummaryWriter
from utils import Vocabulary, TradDataset, collate_pad
logging.basicConfig(level=logging.INFO)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    FILE = "data/en-fra.txt"
    writer = SummaryWriter()

    with open(FILE) as f:
        lines = f.readlines()

    MAX_LEN = 40

    torch.manual_seed(5)

    lines = [lines[x] for x in torch.randperm(len(lines))]
    idxTrain = int(0.9 * len(lines))

    vocEng = Vocabulary(True)
    vocFra = Vocabulary(True)
    print("Loading Data")
    datatrain = TradDataset("".join(lines[:idxTrain]), vocEng, vocFra, spiece=None, max_len=MAX_LEN)
    datatest = TradDataset("".join(lines[idxTrain:]), vocEng, vocFra, spiece=None, max_len=MAX_LEN)
    print(len(vocEng))
    print(len(vocFra))
    INPUT_DIM = len(vocEng)
    OUTPUT_DIM = len(vocFra)
    LATENT_DIM = 150  # 100
    EMBED_DIM = 150
    N_LAYERS = 1
    PACK = True
    BATCH_SIZE = 128
    N_EPOCHS = 100  # 500
    LR = 5e-3
    SAVE = True

    train_loader = DataLoader(datatrain, collate_fn=collate_pad(0), batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    test_loader = DataLoader(datatest, collate_fn=collate_pad(0), batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

    teacher_forcing_probs = [.75] * 25 + [.5] * 40 + [.25] * 25 + [0.] * 10

    encoder = Encoder(INPUT_DIM, EMBED_DIM, LATENT_DIM, N_LAYERS, PACK, device, 0)
    decoder = Decoder(OUTPUT_DIM, EMBED_DIM, LATENT_DIM, N_LAYERS, PACK, device, 0)
    params = [encoder.parameters(), decoder.parameters()]

    optimizer = torch.optim.Adam(chain(*params), lr=LR)
    crossentropy = nn.CrossEntropyLoss(ignore_index=0)

    loss_trace_train = []
    loss_trace_test = []
    print("Start Training")
    for epoch in range(N_EPOCHS):
        epoch_loss_train = 0
        for n, data in enumerate(tqdm(train_loader)):
            timestep = (n + epoch * (len(train_loader.dataset) // BATCH_SIZE)) * BATCH_SIZE
            src, len_src, trgt, len_trgt = data
            src = src.to(device)
            trgt = trgt.to(device)

            # Replacing 1 word by OOV at random in each sequence
            n_oov = torch.randint(0, 1, (1,)).item() + 1
            oov_idx = torch.stack([torch.randint(0, len_src[i].item(), (n_oov,)) for i in range(src.shape[1])]).t()
            src = src.scatter(0, oov_idx.to(device), Vocabulary.OOVID)

            h_last_layer, hidden = encoder(src, len_src)
            if torch.rand(1).item() < teacher_forcing_probs[epoch]:
                preds, h = decoder(trgt[:-1], hidden)
            else:
                preds, h = decoder.generate(hidden, lenseq=len_trgt.max()-1)

            loss = crossentropy(preds.flatten(end_dim=1), trgt[:-1].flatten(end_dim=1))
            writer.add_scalar("Loss Train", loss.item(), timestep)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss_train += loss.item() * src.shape[1]

            clip_grad_norm_(encoder.parameters(), 1)
            clip_grad_norm_(decoder.parameters(), 1)
            params_encoder = list(encoder.parameters())
            params_decoder = list(decoder.parameters())
            encoder_grad_norm = np.mean([torch.norm(param_encoder.grad).item() for param_encoder in params_encoder])
            decoder_grad_norm = np.mean([torch.norm(param_decoder.grad).item() for param_decoder in params_decoder])
            writer.add_scalar("Grad/Encoder", encoder_grad_norm, timestep)
            writer.add_scalar("Grad/Decoder", decoder_grad_norm, timestep)

        epoch_loss_train /= len(train_loader.dataset)
        loss_trace_train.append(epoch_loss_train)
        # print(torch.cuda.memory_summary(device))
        epoch_loss_test = 0
        for n, data in enumerate(tqdm(test_loader)):
            timestep = (n + epoch * (len(test_loader.dataset) // BATCH_SIZE)) * BATCH_SIZE
            src, len_src, trgt, len_trgt = data
            src = src.to(device)
            trgt = trgt.to(device)

            with torch.no_grad():
                h_last_layer, hidden = encoder(src, len_src)
                preds, h = decoder.generate(hidden, lenseq=len_trgt.max())
                loss = crossentropy(preds.flatten(end_dim=1), trgt.flatten(end_dim=1))
                epoch_loss_test += loss.item() * src.shape[1]
                writer.add_scalar("Loss Test", loss.item(), timestep)
        epoch_loss_test /= len(test_loader.dataset)
        loss_trace_test.append(epoch_loss_test)

        print(f"Epoch {epoch}, Loss Train : {epoch_loss_train}, Loss Test : {epoch_loss_test}")
        print(" ".join(vocEng.getwords(src[:len_src[0], 0])))
        print(" ".join(vocFra.getwords(preds.argmax(-1)[:len_trgt[0], 0])))
        print(" ".join(vocFra.getwords(trgt[:len_trgt[0], 0])))

        if SAVE and (epoch % 10 == 0 or epoch == N_EPOCHS - 1) and (epoch > 0):
            torch.save({'epoch': epoch,
                        'encoder_state_dict': encoder.state_dict(),
                        'decoder_state_dict': decoder.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss_train': epoch_loss_train,
                        'loss_trace_train': loss_trace_train,
                        'loss_test': epoch_loss_test,
                        'loss_trace_test': loss_trace_test,
                        'vocEng': vocEng,
                        'vocFra': vocFra,
                        },
                       f"models/seq2seq_traduction_maxlen_{MAX_LEN}_emb{EMBED_DIM}_latent{LATENT_DIM}_nlayers{N_LAYERS}.pth")
