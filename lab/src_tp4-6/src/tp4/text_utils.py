import string
import unicodedata
import torch
from torch.utils.data import Dataset

eos_code = 0
eos = "|"
LETTRES = eos + string.ascii_letters + string.punctuation.replace(eos, "") + string.digits + ' '
id2lettre = dict(zip(range(len(LETTRES)), LETTRES))
id2lettre[eos_code] = eos
lettre2id = dict(zip(id2lettre.values(), id2lettre.keys()))
VOCAB_SIZE = len(id2lettre)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def normalize(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if c in LETTRES)


def string2code(s):
    return torch.tensor([lettre2id[c] for c in normalize(s)])


def code2string(t):
    if type(t) != list:
        t = t.tolist()
    return ''.join(id2lettre[i] for i in t)


def collate_fn(samples):
    # Pick up  the minimum length of sentences in the batch
    min_seq_len = min([sample.shape[0] for sample in samples])
    # dimension returned seq_len, batch_size, dim
    list_sub_seqs = []
    for seq in samples:
        sub_seq = seq[:min_seq_len]
        # sub_seq[-1] = EOS_IX
        list_sub_seqs.append(sub_seq)
    samples = torch.stack(list_sub_seqs).permute(1, 0)
    return samples


class TextDataset(Dataset):
    def __init__(self, text: str, maxsent=None, maxlen=None, minlen=0):
        self.text = text
        self.maxsent = maxsent
        self.maxlen = maxlen
        self.minlen = minlen
        self.nchars = len(text)
        self.sentences = [s.strip() for s in text.split(". ") if len(s.strip()) > self.minlen]
        self.sentences = self.sentences

    def __len__(self):
        return self.sentences.__len__()  # len(self.text) - self.maxlen

    def __getitem__(self, i):
        # code_sentence = string2code(self.text[i:i + self.maxlen])
        code_sentence = string2code(self.sentences[i])
        # Add EOS at the end
        return code_sentence  # torch.cat([code_sentence, torch.tensor(EOS_IX).unsqueeze(0)])



