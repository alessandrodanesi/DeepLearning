import numpy as np
import torch
from torch.utils.data import Dataset
from functools import partial
from pathlib import Path
from datamaestro import prepare_dataset
from torch.nn.utils.rnn import pad_sequence
import re
import logging


class FolderText(Dataset):
    """Dataset basé sur des dossiers (un par classe) et fichiers"""

    def __init__(self, classes, folder: Path, tokenizer, load=False):
        self.tokenizer = tokenizer
        self.files = []
        self.filelabels = []
        self.labels = {}
        for ix, key in enumerate(classes):
            self.labels[key] = ix

        for label in classes:
            for file in (folder / label).glob("*.txt"):
                self.files.append(file.read_text(encoding="utf-8") if load else file)
                self.filelabels.append(self.labels[label])

    def __len__(self):
        return len(self.filelabels)

    def __getitem__(self, ix):
        s = self.files[ix]
        return self.tokenizer(s if isinstance(s, str) else s.read_text(encoding="utf-8")), self.filelabels[ix]


def get_imdb_data(embedding_size=50):
    """Renvoie l'ensemble des donnéees nécessaires pour l'apprentissage

    - dictionnaire word vers ID
    - embeddings (Glove)
    - DataSet (FolderText)

    """
    WORDS = re.compile(r"\S+")
    words, embeddings = prepare_dataset('edu.stanford.glove.6b.%d' % embedding_size).load()
    OOVID = len(words)
    words.append("__OOV__")
    PADID = len(words)
    words.append("__PAD__")
    CLSID = len(words)
    words.append("__CLS__")

    word2id = {word: ix for ix, word in enumerate(words)}
    embeddings = np.vstack((embeddings, np.zeros(embedding_size), np.zeros(embedding_size), np.zeros(embedding_size)))

    def tokenizer(t):
        return [word2id.get(x, OOVID) for x in re.findall(WORDS, t.lower())]

    logging.info("Loading embeddings")

    logging.info("Get the IMDB dataset")
    ds = prepare_dataset("edu.stanford.aclimdb")
    train = FolderText(ds.train.classes, ds.train.path, tokenizer, load=False)
    test = FolderText(ds.test.classes, ds.test.path, tokenizer, load=False)
    return word2id, embeddings, train, test


def collate_pad(pad, cls=None):
    def collate(batch):
        sentences, labels = zip(*batch)
        if cls:
            sentences = [torch.cat([torch.LongTensor([cls]), torch.LongTensor(sentence)]) for sentence in sentences]
        else:
            sentences = [torch.LongTensor(sentence) for sentence in sentences]
        # sentences_len = torch.tensor([len(sentence) for sentence in sentences])
        labels = torch.LongTensor(labels)
        # return pad_sequence(sentences, padding_value=pad), sentences_len, labels
        return pad_sequence(sentences, padding_value=pad), labels

    return collate
