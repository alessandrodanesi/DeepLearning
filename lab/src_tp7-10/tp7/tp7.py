import logging
import heapq
from pathlib import Path
import gzip
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader, random_split
import sentencepiece as spm
from tp7_preprocess import TextDataset
from cnn import CNNSentiment, CNNSentiment1, CNNSentiment3, test_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def loaddata(mode, voc_size):
    with gzip.open(f"{mode}-{voc_size}.pth", "rb") as fp:
        return torch.load(fp)


if __name__ == "__main__":
    # --- Configuration
    # Taille du vocabulaire
    vocab_size = 1000

    # Chargement du tokenizer
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.Load(f"wp{vocab_size}.model")
    ntokens = len(tokenizer)

    test = loaddata("test", vocab_size)
    train = loaddata("train", vocab_size)
    BATCH_SIZE = 10000
    TEST_BATCHSIZE = 1000

    # --- Chargements des jeux de données train, validation et test
    val_size = 1000
    train_size = len(train) - val_size
    train, val = random_split(train, [train_size, val_size])

    train_loader = DataLoader(train, batch_size=BATCH_SIZE, collate_fn=TextDataset.collate,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val, batch_size=TEST_BATCHSIZE, collate_fn=TextDataset.collate,
                            num_workers=4, pin_memory=True)
    test_loader = DataLoader(test, batch_size=TEST_BATCHSIZE, collate_fn=TextDataset.collate,
                             num_workers=4, pin_memory=True)

    SAVE = True
    N_EPOCHS = 20
    LR = 1e-3

    # model = CNNSentiment(ntokens).to(device)
    # model = CNNSentiment1(ntokens).to(device)
    model = CNNSentiment3(ntokens).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LR)

    writer = SummaryWriter()
    print("Start Training")
    loss_trace_train = []
    loss_trace_test = []
    acc_trace_train = []
    acc_trace_test = []
    for epoch in range(N_EPOCHS):
        epoch_loss_train = 0
        epoch_acc_train = 0
        for x, y in tqdm(train_loader):
            x, y = x.to(device), y.to(device)
            output, _ = model(x)
            loss = criterion(output, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss_train += loss.item() * x.shape[0]
            epoch_acc_train += torch.eq(torch.argmax(output, dim=-1), y).sum().item()

        epoch_loss_train /= len(train_loader.dataset)
        epoch_acc_train /= len(train_loader.dataset)
        loss_trace_train.append(epoch_loss_train)
        acc_trace_train.append(epoch_acc_train)

        # validation
        epoch_loss_test = 0
        epoch_acc_test = 0
        if epoch % 5 == 0 or epoch == N_EPOCHS - 1:
            with torch.no_grad():
                for x, y in tqdm(val_loader):
                    x, y = x.to(device), y.to(device)
                    output, _ = model(x)
                    loss = criterion(output, y)
                    epoch_loss_test += loss.item() * x.shape[0]
                    epoch_acc_test += torch.eq(torch.argmax(output, dim=-1), y).sum().item()

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
        if SAVE and (epoch % 5 == 0 or epoch == N_EPOCHS - 1):
            test_model(model, tokenizer)
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss_trace_train': loss_trace_train,
                        'loss_trace_test': loss_trace_test,
                        'acc_trace_train': acc_trace_train,
                        'acc_trace_test': acc_trace_test,
                        },
                       f"models/{model.__class__.__name__}.pth")
