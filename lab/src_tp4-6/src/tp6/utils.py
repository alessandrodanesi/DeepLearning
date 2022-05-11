import logging
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch
import unicodedata
from string import ascii_letters, punctuation
from tqdm import tqdm
from typing import List
import re
logging.basicConfig(level=logging.INFO)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def normalize(s):
    s = unicodedata.normalize('NFD', s.lower().strip())
    list_s = [c if c in ascii_letters else " " for c in s if c in ascii_letters + " " + punctuation]
    s = "".join(list_s)
    s = re.sub(' +', ' ', s)
    return s.strip()


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
    PAD = 0
    EOS = 1
    SOS = 2
    OOVID = 3

    def __init__(self, oov: bool):
        self.oov = oov
        self.id2word = ["PAD", "EOS", "SOS"]
        self.word2id = {"PAD": Vocabulary.PAD, "EOS": Vocabulary.EOS, "SOS": Vocabulary.SOS}
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


class TradDataset(Dataset):
    def __init__(self, inputdata, vocOrig=None, vocDest=None, spiece=None, adding=True, max_len=20):
        self.sentences = []
        for s in tqdm(inputdata.split("\n")):
            if len(s) < 1:
                continue
            orig, dest = map(normalize, s.split("\t")[:2])
            if len(orig) > max_len:
                continue
            if spiece:
                org = torch.tensor(spiece.encode(orig, out_type=int, add_bos=True, add_eos=True))
                dest = torch.tensor(spiece.encode(dest, out_type=int, add_bos=True, add_eos=True))
            else:
                org = torch.tensor(
                    [Vocabulary.SOS] + [vocOrig.get(o, adding) for o in orig.split(" ")] + [Vocabulary.EOS])
                dest = torch.tensor(
                    [Vocabulary.SOS] + [vocDest.get(o, adding) for o in dest.split(" ")] + [Vocabulary.EOS])
            self.sentences.append((org, dest))

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, i):
        return self.sentences[i]


def collate_pad(pad):
    def collate(batch):
        orig, dest = zip(*batch)
        o_len = torch.tensor([len(o) for o in orig])
        d_len = torch.tensor([len(d) for d in dest])
        return pad_sequence(orig, padding_value=pad), o_len, pad_sequence(dest, padding_value=pad), d_len
    return collate


def prepare_str(sentence: str, vocOrig):
    sentence = normalize(sentence)
    org = torch.tensor([Vocabulary.SOS] + [vocOrig.get(o, False) for o in sentence.split(" ")] + [Vocabulary.EOS])
    o_len = torch.tensor([len(org)])
    return pad_sequence(org), o_len
