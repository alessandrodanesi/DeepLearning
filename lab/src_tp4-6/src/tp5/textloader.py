from torch.utils.data import Dataset, DataLoader
import unicodedata
import string
import re
import torch

pad_ix = 0
eos_ix = 1
eos = "|"
pad = ""  # NULL CHARACTER

LETTRES = eos + string.ascii_letters + string.punctuation.replace(eos, "") + string.digits + ' '
id2lettre = dict(zip(range(1, len(LETTRES) + 1), LETTRES))
id2lettre[pad_ix] = pad
# id2lettre[eos_ix] = eos
lettre2id = dict(zip(id2lettre.values(), id2lettre.keys()))


def normalize(s):
    """ enlève les accents et les majuscules """
    return ''.join(c for c in unicodedata.normalize('NFD', s) if c in LETTRES)


def string2code(s):
    """prend une séquence de lettres et renvoie la séquence d'entiers correspondantes"""
    return torch.tensor([lettre2id[c] for c in normalize(s)])


def code2string(t):
    """ prend une séquence d'entiers et renvoie la séquence de lettres correspondantes """
    if not isinstance(t, list):
        t = t.tolist()
    return ''.join(id2lettre[i] for i in t)


class TextDataset(Dataset):
    def __init__(self, text: str, maxsent=None, maxlen=None, minlen=5):
        self.text = text
        self.maxsent = maxsent
        self.maxlen = maxlen
        self.minlen = minlen
        self.sentences = [s.replace(eos, "") for s in re.split(r'[-]{2}|[?!.]( |$)', self.text) if s]
        self.sentences = [s.strip() for s in self.sentences if len(s.strip()) > self.minlen]
        self.sentences = self.sentences[:self.maxsent]

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, i):
        # Add EOS at the end
        return torch.cat([string2code(self.sentences[i][:self.maxlen]), torch.tensor([eos_ix])])


def pad_seq(seq, max_len: int):
    seq_len = seq.shape[0]
    padd_num = max_len - seq_len
    if padd_num >= 0:
        padd = torch.ones(padd_num) * pad_ix
        seq = torch.cat([seq, padd])
    return seq


def collate_fn(samples):
    max_len = max(map(len, samples))
    samples = [pad_seq(seq, max_len) for seq in samples]
    return torch.stack(samples).permute(1, 0)


if __name__ == "__main__":
    test = " C'est .  Un .  Test .  Il faut des séquences de tailes différentes ! Des questions ?"
    ds = TextDataset(test, minlen=5)
    print(ds[0])
    print(pad_seq(ds[0], 100))
    loader = DataLoader(ds, collate_fn=collate_fn, batch_size=3)
    data = next(iter(loader))
    print(data)
    from torch import nn
    criterion = nn.CrossEntropyLoss(reduction="none")
    y_pred = torch.randn(data.shape[0], data.shape[1], len(LETTRES)+2)+10
    print(y_pred.flatten(end_dim=1).shape)
    print(data.flatten(end_dim=1).shape)
    print(criterion(y_pred.flatten(end_dim=1), data.long().flatten(end_dim=1)) * (data == pad_ix).flatten(end_dim=1))
    print((data == pad_ix).flatten(end_dim=1))
