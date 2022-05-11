import torch
from torch import nn
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def receptive_field(model, ind):
    w = 0
    s = 1
    for name, m in model.named_modules():
        if "conv" in name:
            w += (m.kernel_size[0]-1) * m.stride[0]
            s *= m.stride[0]
    return ind * s, ind * s + w


def test_model(model, tokenizer):
    with torch.no_grad():
        test_string = "Flowers are blue. Roses are red. Flowers are red sometimes. Blue birds."
        test_ids = torch.LongTensor(tokenizer.encode(test_string)).unsqueeze(0).to(device)
        output, _ = model(test_ids)
        output = output.squeeze(0)
        print(f"Test string: {test_string}")
        print(output.argmax(-1))

        test_string = "This movie was not bad at all. Liked it a lot."
        test_ids = torch.LongTensor(tokenizer.encode(test_string)).unsqueeze(0).to(device)
        output, _ = model(test_ids)
        output = output.squeeze(0)
        print(f"Test string: {test_string}")
        print(output.argmax(-1))

        test_string = "This movie was anoying. Did not hate it, but it was bad!"
        test_ids = torch.LongTensor(tokenizer.encode(test_string)).unsqueeze(0).to(device)
        output, _ = model(test_ids)
        output = output.squeeze(0)
        print(f"Test string: {test_string}")
        print(output.argmax(-1))


class CNNSentiment1(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        embed_dim = 64
        nfilters = 32
        self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv1 = nn.Conv1d(embed_dim, nfilters, 3, stride=1, padding=1)
        self.maxpool1 = nn.MaxPool1d(3, ceil_mode=True)
        self.adamaxpool = nn.AdaptiveMaxPool1d(1, return_indices=True)

        self.classifier = nn.Linear(nfilters, 2)

    def forward(self, x):
        out = self.emb(x).permute(0, 2, 1)
        out = self.maxpool1(F.relu(self.conv1(out)))
        out, idx = self.adamaxpool(out)
        out = self.classifier(out.view(x.size(0), -1))
        return out, idx


class CNNSentiment(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        embed_dim = 64
        nfilters = 32
        self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv1 = nn.Conv1d(embed_dim, nfilters, 3, stride=1, padding=1)
        self.maxpool1 = nn.MaxPool1d(3)
        self.conv2 = nn.Conv1d(nfilters, 2 * nfilters, 3, stride=1, padding=1)
        self.maxpool2 = nn.MaxPool1d(3, ceil_mode=True)
        self.adamaxpool = nn.AdaptiveMaxPool1d(1, return_indices=True)

        self.classifier = nn.Linear(2 * nfilters, 2)

    def forward(self, x):
        out = self.emb(x).permute(0, 2, 1)
        out = self.maxpool1(F.relu(self.conv1(out)))
        out = self.maxpool2(F.relu(self.conv2(out)))
        out, idx = self.adamaxpool(out)
        out = self.classifier(out.view(x.size(0), -1))
        return out, idx


class CNNSentiment3(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        embed_dim = 64
        nfilters = 32
        self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv1 = nn.Conv1d(embed_dim, nfilters, 3, stride=1, padding=1)
        self.maxpool1 = nn.MaxPool1d(3)
        self.conv2 = nn.Conv1d(nfilters, 2 * nfilters, 3, stride=1, padding=1)
        self.maxpool2 = nn.MaxPool1d(3)
        self.conv3 = nn.Conv1d(2 * nfilters, 4 * nfilters, 3, stride=1, padding=1)
        self.maxpool3 = nn.MaxPool1d(3, ceil_mode=True)
        self.adamaxpool = nn.AdaptiveMaxPool1d(1, return_indices=True)

        self.classifier = nn.Linear(4 * nfilters, 2)

    def forward(self, x):
        out = self.emb(x).permute(0, 2, 1)
        out = self.maxpool1(F.relu(self.conv1(out)))
        out = self.maxpool2(F.relu(self.conv2(out)))
        out = self.maxpool3(F.relu(self.conv3(out)))
        out, idx = self.adamaxpool(out)
        out = self.classifier(out.view(x.size(0), -1))
        return out, idx