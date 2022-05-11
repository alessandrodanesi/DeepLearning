import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data_utils import get_imdb_data, collate_pad
from tqdm import tqdm
from attention import AttentionModelQ3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    # ### Parameters ### #
    N_EPOCHS = 20
    LR = 1e-3
    BATCH_SIZE = 16
    EMB_DIM = 50
    OUT_DIM = 2
    SAVE = True

    print("Loading Data")
    word2id, embeddings, train, test = get_imdb_data(embedding_size=EMB_DIM)
    train_loader = DataLoader(train, collate_fn=collate_pad(word2id["__PAD__"]), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test, collate_fn=collate_pad(word2id["__PAD__"]), batch_size=BATCH_SIZE, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    model = AttentionModelQ3(EMB_DIM, OUT_DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    loss_trace_train = []
    loss_trace_test = []
    acc_trace_train = []
    acc_trace_test = []
    print("Start Training")
    for epoch in range(N_EPOCHS):
        # training
        epoch_loss_train = 0
        epoch_acc_train = 0
        for batch in tqdm(train_loader):
            sentences, labels = batch
            mask_pad = (sentences == word2id["__PAD__"]).to(device)
            x_emb = torch.Tensor(embeddings[sentences]).to(device)
            labels = labels.to(device)
            y_hat, att = model(x_emb, mask_pad)
            y_hat = y_hat.squeeze(0)
            optimizer.zero_grad()
            loss = criterion(y_hat, labels)
            loss.backward()
            optimizer.step()
            epoch_loss_train += loss.item() * sentences.shape[1]
            epoch_acc_train += torch.eq(torch.argmax(y_hat, dim=-1), labels).sum().item()

        epoch_loss_train /= len(train_loader.dataset)
        epoch_acc_train /= len(train_loader.dataset)
        loss_trace_train.append(epoch_loss_train)
        acc_trace_train.append(epoch_acc_train)
        # validation
        epoch_loss_test = 0
        epoch_acc_test = 0
        with torch.no_grad():
            for batch in tqdm(test_loader):
                sentences, labels = batch
                mask_pad = (sentences == word2id["__PAD__"]).to(device)
                x_emb = torch.Tensor(embeddings[sentences]).to(device)
                labels = labels.to(device)
                y_hat, att = model(x_emb, mask_pad)
                y_hat = y_hat.squeeze(0)
                loss = criterion(y_hat, labels)
                epoch_loss_test += loss.item() * sentences.shape[1]
                epoch_acc_test += torch.eq(torch.argmax(y_hat, dim=-1), labels).sum().item()

            epoch_loss_test /= len(test_loader.dataset)
            epoch_acc_test /= len(test_loader.dataset)
            loss_trace_test.append(epoch_loss_test)
            acc_trace_test.append(epoch_acc_test)

        print(f"Epoch {epoch}, Loss Train : {epoch_loss_train}, Loss Test : {epoch_loss_test}")
        print(f"Epoch {epoch}, Acc Train : {epoch_acc_train}, Acc Test : {epoch_acc_test}")

        # checkpoint
        if SAVE and (epoch % 5 == 0 or epoch == N_EPOCHS - 1) and (epoch > 0):
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss_trace_train': loss_trace_train,
                        'loss_trace_test': loss_trace_test,
                        'acc_trace_train': acc_trace_train,
                        'acc_trace_test': acc_trace_test,
                        },
                       f"models/attention_model_q3_bsize{BATCH_SIZE}.pth")
