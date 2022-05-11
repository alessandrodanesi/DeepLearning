import logging
from tqdm import tqdm

from datamaestro import prepare_dataset
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_sequence, pack_padded_sequence, pad_packed_sequence
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch
from typing import List
import time
from itertools import chain
from seq2seq import EncoderTagging
logging.basicConfig(level=logging.INFO)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Vocabulary:
    """Permet de gérer un vocabulaire.

    En test, il est possible qu'un mot ne soit pas dans le
    vocabulaire : dans ce cas le token "__OOV__" est utilisé.
    Attention : il faut tenir compte de cela lors de l'apprentissage !

    Utilisation:

    - en train, utiliser v.get("blah", adding=True) pour que le mot soit ajouté
      automatiquement
    - en test, utiliser v["blah"] pour récupérer l'ID du mot (ou l'ID de OOV)
    """
    OOVID = 1
    PAD = 0

    def __init__(self, oov: bool):
        self.oov = oov
        self.id2word = ["PAD"]
        self.word2id = {"PAD": Vocabulary.PAD}
        if oov:
            self.word2id["__OOV__"] = Vocabulary.OOVID
            self.id2word.append("__OOV__")

    def __getitem__(self, word: str):
        if self.oov:
            return self.word2id.get(word, Vocabulary.OOVID)
        return self.word2id[word]

    def get(self, word: str, adding=True):
        try:
            return self.word2id[word]
        except KeyError:
            if adding:
                wordid = len(self.id2word)
                self.word2id[word] = wordid
                self.id2word.append(word)
                return wordid
            if self.oov:
                return Vocabulary.OOVID
            raise

    def __len__(self):
        return len(self.id2word)

    def getword(self, idx: int):
        if idx < len(self):
            return self.id2word[idx]
        return None

    def getwords(self, idx: List[int]):
        return [self.getword(i) for i in idx]


class TaggingDataset(Dataset):
    def __init__(self, data, words: Vocabulary, tags: Vocabulary, adding=True):
        self.sentences = []

        for s in data:
            self.sentences.append(([words.get(token["form"], adding) for token in s],
                                   [tags.get(token["upostag"], adding) for token in s]))

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, ix):
        return self.sentences[ix]


# def collate(batch):
#     """Collate using pad_sequence"""
#     return tuple(pad_sequence([torch.LongTensor(b[j]) for b in batch]) for j in range(2))
#

def collate(batch):
    """Collate using pad_sequence"""
    input_seqs, target_tags = zip(*batch)
    seqs_len = torch.tensor([len(s) for s in input_seqs])
    pad_input_seqs = pad_sequence([torch.LongTensor(s) for s in input_seqs])
    pad_target_tags = pad_sequence([torch.LongTensor(t) for t in target_tags])
    return pad_input_seqs, seqs_len, pad_target_tags


if __name__ == "__main__":
    ds = prepare_dataset('org.universaldependencies.french.gsd')  # Format de sortie décrit dans
    # https://pypi.org/project/conllu/

    logging.info("Loading datasets...")
    words = Vocabulary(True)
    tags = Vocabulary(False)
    train_data = TaggingDataset(ds.train, words, tags, True)
    dev_data = TaggingDataset(ds.validation, words, tags, True)
    test_data = TaggingDataset(ds.test, words, tags, False)

    logging.info("Vocabulary size: %d", len(words))

    INPUT_DIM = len(words)
    OUTPUT_DIM = len(tags)
    LATENT_DIM = 200
    EMBED_DIM = 200
    N_LAYERS = 1
    PACK = True
    BATCH_SIZE = 300
    N_EPOCHS = 50
    LR = 1e-3
    SAVE = True

    train_loader = DataLoader(train_data, collate_fn=collate, batch_size=BATCH_SIZE, shuffle=True)
    dev_loader = DataLoader(dev_data, collate_fn=collate, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_data, collate_fn=collate, batch_size=BATCH_SIZE)

    encoder = EncoderTagging(INPUT_DIM, EMBED_DIM, LATENT_DIM, N_LAYERS, PACK, device)
    decoder = nn.Linear(LATENT_DIM, OUTPUT_DIM).to(device)
    params = [encoder.parameters(), decoder.parameters()]

    optimizer = torch.optim.Adam(chain(*params), lr=LR)
    crossentropy = nn.CrossEntropyLoss(ignore_index=0)

    loss_trace_train = []
    loss_trace_test = []
    for epoch in range(N_EPOCHS):
        epoch_loss_train = 0
        for n, data in enumerate(tqdm(train_loader)):
            timestep = (n + epoch * (len(train_loader.dataset) // BATCH_SIZE)) * BATCH_SIZE
            seqs, lengths, targets = data
            seqs = seqs.to(device)
            targets = targets.to(device)

            # Replacing up to 3 words by OOV at random in each sequence
            n_oov = torch.randint(0, 3, (1,)).item() + 1
            oov_idx = torch.stack([torch.randint(0, lengths[i].item(), (n_oov, )) for i in range(seqs.shape[1])]).t()
            seqs = seqs.scatter(0, oov_idx.to(device), Vocabulary.OOVID)

            h_last_layer, (h_last, c_last) = encoder(seqs, lengths)
            preds = torch.stack([decoder(h) for h in h_last_layer])

            loss = crossentropy(preds.flatten(end_dim=1), targets.flatten(end_dim=1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss_train += loss.item() * seqs.shape[1]
        epoch_loss_train /= len(train_loader.dataset)
        loss_trace_train.append(epoch_loss_train)

        epoch_loss_test = 0
        for n, data in enumerate(tqdm(dev_loader)):
            timestep = (n + epoch * (len(dev_loader.dataset) // BATCH_SIZE)) * BATCH_SIZE
            seqs, lengths, targets = data
            seqs = seqs.to(device)
            targets = targets.to(device)

            with torch.no_grad():
                h_last_layer, (h_last, c_last) = encoder(seqs, lengths)
                preds = torch.stack([decoder(h) for h in h_last_layer])
                loss = crossentropy(preds.flatten(end_dim=1), targets.flatten(end_dim=1))
                epoch_loss_test += loss.item() * seqs.shape[1]
        epoch_loss_test /= len(dev_loader.dataset)
        loss_trace_test.append(epoch_loss_test)

        print(f"Epoch {epoch}, Loss Train : {epoch_loss_train}, Loss Test : {epoch_loss_test}")
        if epoch % 10 == 0 and (epoch > 0):
            with torch.no_grad():
                # Test Tagging
                # Input sentence
                k = torch.randint(0, 15, (1,)).item()
                seq = seqs[:, k]
                pred = preds[:, k].argmax(dim=-1)
                target = targets[:, k]
                try:
                    T = torch.where(seq.eq(0))[0][0].item()
                except IndexError:
                    T = None
                print(' '.join(words.getwords(seq[:T])))
                print(' '.join(tags.getwords(pred[:T])))
                print(' '.join(tags.getwords(target[:T])))

        if SAVE and (epoch % 20 == 0 or epoch == N_EPOCHS - 1) and (epoch > 0):
            torch.save({'epoch': epoch,
                        'encoder_state_dict': encoder.state_dict(),
                        'decoder_state_dict': decoder.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss_train': epoch_loss_train,
                        'loss_trace_train': loss_trace_train,
                        'loss_test': epoch_loss_test,
                        'loss_trace_test': loss_trace_test
                        },
                       f"models/seq2seq_tagging_emb{EMBED_DIM}_latent{LATENT_DIM}_nlayers{N_LAYERS}.pth")