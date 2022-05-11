import logging
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch
from tqdm import tqdm
from seq2seq import Encoder, Decoder
import numpy as np
from itertools import chain
from torch.utils.tensorboard import SummaryWriter
import sentencepiece as spm
from utils import TradDataset, collate_pad

logging.basicConfig(level=logging.INFO)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    FILE = "data/en-fra.txt"
    writer = SummaryWriter()

    with open(FILE) as f:
        lines = f.readlines()

    lines = [lines[x] for x in torch.randperm(len(lines))]
    idxTrain = int(0.9 * len(lines))

    torch.manual_seed(5)
    MAX_LEN = 40
    VOCAB_SIZE = 10000

    PAD_ID = 0
    EOS_ID = 1
    SOS_ID = 2
    OOV_ID = 3
    train_sentencepiece = False
    if train_sentencepiece:
        spm.SentencePieceTrainer.train(input="data/en-fra.txt", model_prefix=f"models/spiece_vocabsize{VOCAB_SIZE}",
                                       vocab_size=VOCAB_SIZE, pad_id=0, eos_id=1, bos_id=2, unk_id=3)
    sp = spm.SentencePieceProcessor()
    sp.load(f'models/spiece_vocabsize{VOCAB_SIZE}.model')

    datatrain = TradDataset("".join(lines[:idxTrain]), spiece=sp, max_len=MAX_LEN)
    datatest = TradDataset("".join(lines[idxTrain:]), spiece=sp, max_len=MAX_LEN)

    INPUT_DIM = VOCAB_SIZE
    OUTPUT_DIM = VOCAB_SIZE
    LATENT_DIM = 200  # 100
    EMBED_DIM = 100
    N_LAYERS = 1
    PACK = True
    BATCH_SIZE = 128
    N_EPOCHS = 100  # 500
    LR = 4e-3
    SAVE = True

    train_loader = DataLoader(datatrain, collate_fn=collate_pad(PAD_ID), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(datatest, collate_fn=collate_pad(PAD_ID), batch_size=BATCH_SIZE, shuffle=True)

    # teacher_forcing_probs = torch.logspace(0, -1, steps=N_EPOCHS)
    teacher_forcing_probs = [.5] * 25 + [.5] * 25 + [.25] * 25 + [0.] * 25

    encoder = Encoder(INPUT_DIM, EMBED_DIM, LATENT_DIM, N_LAYERS, PACK, device, PAD_ID)
    decoder = Decoder(OUTPUT_DIM, EMBED_DIM, LATENT_DIM, N_LAYERS, PACK, device, PAD_ID)
    params = [encoder.parameters(), decoder.parameters()]

    optimizer = torch.optim.Adam(chain(*params), lr=LR)
    crossentropy = nn.CrossEntropyLoss(ignore_index=PAD_ID)

    loss_trace_train = []
    loss_trace_test = []
    for epoch in range(N_EPOCHS):
        epoch_loss_train = 0
        for n, data in enumerate(tqdm(train_loader)):
            timestep = (n + epoch * (len(train_loader.dataset) // BATCH_SIZE)) * BATCH_SIZE
            src, len_src, trgt, len_trgt = data
            src = src.to(device)
            trgt = trgt.to(device)

            h_last_layer, hidden = encoder(src, len_src)
            if torch.rand(1).item() < teacher_forcing_probs[epoch]:
                preds, h = decoder(trgt[:-1], hidden)
            else:
                preds, h = decoder.generate(hidden, lenseq=len_trgt.max()-1, sos=SOS_ID, eos=EOS_ID, pad=PAD_ID)

            loss = crossentropy(preds.flatten(end_dim=1), trgt[:-1].flatten(end_dim=1))
            writer.add_scalar("Loss Train", loss.item(), timestep)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss_train += loss.item() * src.shape[1]

            params_encoder = list(encoder.parameters())
            params_decoder = list(decoder.parameters())
            encoder_grad_norm = np.mean([torch.norm(param_encoder.grad).item() for param_encoder in params_encoder])
            decoder_grad_norm = np.mean([torch.norm(param_decoder.grad).item() for param_decoder in params_decoder])
            writer.add_scalar("Grad/Encoder", encoder_grad_norm, timestep)
            writer.add_scalar("Grad/Decoder", decoder_grad_norm, timestep)

        epoch_loss_train /= len(train_loader.dataset)
        loss_trace_train.append(epoch_loss_train)

        epoch_loss_test = 0
        for n, data in enumerate(tqdm(test_loader)):
            timestep = (n + epoch * (len(test_loader.dataset) // BATCH_SIZE)) * BATCH_SIZE
            src, len_src, trgt, len_trgt = data
            src = src.to(device)
            trgt = trgt.to(device)

            with torch.no_grad():
                h_last_layer, hidden = encoder(src, len_src)
                preds, h = decoder.generate(hidden, lenseq=len_trgt.max(), sos=SOS_ID, eos=EOS_ID, pad=PAD_ID)
                loss = crossentropy(preds.flatten(end_dim=1), trgt.flatten(end_dim=1))
                epoch_loss_test += loss.item() * src.shape[1]
                writer.add_scalar("Loss Test", loss.item(), timestep)
        epoch_loss_test /= len(test_loader.dataset)
        loss_trace_test.append(epoch_loss_test)

        print(f"Epoch {epoch}, Loss Train : {epoch_loss_train}, Loss Test : {epoch_loss_test}")
        print("".join(sp.decode(src[:len_src[0], 0].tolist())))
        print("".join(sp.decode(preds.argmax(-1)[:len_trgt[0], 0].tolist())))
        print("".join(sp.decode(trgt[:len_trgt[0], 0].tolist())))

        if SAVE and (epoch % 20 == 0 or epoch == N_EPOCHS - 1) and (epoch > 0):
            torch.save({'epoch': epoch,
                        'encoder_state_dict': encoder.state_dict(),
                        'decoder_state_dict': decoder.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss_train': epoch_loss_train,
                        'loss_trace_train': loss_trace_train,
                        'loss_test': epoch_loss_test,
                        'loss_trace_test': loss_trace_test,
                        },
                       f"models/seq2seq_traduction_sentpiece_maxlen_{MAX_LEN}_emb{EMBED_DIM}_latent{LATENT_DIM}_nlayers{N_LAYERS}_v4.pth")
