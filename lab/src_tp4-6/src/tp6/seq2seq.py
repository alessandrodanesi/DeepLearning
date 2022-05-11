import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np


class EncoderTagging(nn.Module):
    def __init__(self, input_dim, embed_dim, latent_dim, n_layers, packed_sequences, device):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.n_layers = n_layers
        self.packed_sequences = packed_sequences
        self.device = device
        self.emb = nn.Embedding(input_dim, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, latent_dim, n_layers)

        self.to(device)

    def to(self, device):
        self.device = device
        super().to(device)

    @staticmethod
    def pack_sequences(x_seqs, lengths):
        lengths, sorted_idx = lengths.sort(dim=0, descending=True)  # sorting ids
        _, unsorted_idx = sorted_idx.sort(dim=0)  # unsorting ids
        x_seqs = x_seqs[:, sorted_idx]  # sorting seqs by length
        x_packed = pack_padded_sequence(x_seqs, lengths.cpu())
        return x_packed, unsorted_idx

    @staticmethod
    def unpack_sequences(h_last_layer, h, c, unsorted_idx):
        h_last_layer_padded, h_last_layer_lengths = pad_packed_sequence(h_last_layer)
        h_last_layer = h_last_layer_padded[:, unsorted_idx]
        h = h[:, unsorted_idx]
        c = c[:, unsorted_idx]
        return h_last_layer, (h, c)

    def forward(self, x_seqs, lengths=None):
        x_seqs = self.emb(x_seqs.long())
        if self.packed_sequences and (lengths is not None):
            x_seqs, unsorted_idx = self.pack_sequences(x_seqs, lengths)  # Packing
            h_last_layer, (h, c) = self.lstm(x_seqs)
            h_last_layer, (h, c) = self.unpack_sequences(h_last_layer, h, c, unsorted_idx)  # Unpacking
        else:
            h_last_layer, (h, c) = self.lstm(x_seqs)
        return h_last_layer, (h, c)


class Encoder(nn.Module):
    def __init__(self, input_dim, embed_dim, latent_dim, n_layers, packed_sequences, device, pad_id):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.n_layers = n_layers
        self.packed_sequences = packed_sequences
        self.device = device
        self.emb = nn.Embedding(input_dim, embed_dim, padding_idx=pad_id)
        self.gru = nn.GRU(embed_dim, latent_dim, n_layers)

        self.to(device)

    def to(self, device):
        self.device = device
        super().to(device)

    @staticmethod
    def pack_sequences(x_seqs, lengths):
        lengths, sorted_idx = lengths.sort(dim=0, descending=True)  # sorting ids
        _, unsorted_idx = sorted_idx.sort(dim=0)  # unsorting ids
        x_seqs = x_seqs[:, sorted_idx]  # sorting seqs by length
        x_packed = pack_padded_sequence(x_seqs, lengths.cpu())
        return x_packed, unsorted_idx

    @staticmethod
    def unpack_sequences(h_last_layer, h, unsorted_idx):
        h_last_layer_padded, h_last_layer_lengths = pad_packed_sequence(h_last_layer)
        h_last_layer = h_last_layer_padded[:, unsorted_idx]
        h = h[:, unsorted_idx]
        return h_last_layer, h

    def forward(self, x_seqs, lengths=None):
        x_seqs = self.emb(x_seqs.long())
        if self.packed_sequences and (lengths is not None):
            x_seqs, unsorted_idx = self.pack_sequences(x_seqs, lengths)  # Packing
            h_last_layer, h = self.gru(x_seqs)
            h_last_layer, h = self.unpack_sequences(h_last_layer, h, unsorted_idx)  # Unpacking
        else:
            h_last_layer, h = self.gru(x_seqs)

        return h_last_layer, h


class Decoder(nn.Module):
    def __init__(self, output_dim, embed_dim, latent_dim, n_layers, packed_sequences, device, pad_id):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.packed_sequences = packed_sequences
        self.device = device

        self.emb = nn.Sequential(nn.Embedding(output_dim, embed_dim, padding_idx=pad_id), nn.ReLU())
        self.gru = nn.GRU(embed_dim, latent_dim, n_layers)
        self.head = nn.Linear(latent_dim, output_dim)

        self.to(device)

    def to(self, device):
        self.device = device
        super().to(device)

    @staticmethod
    def pack_sequences(x_seqs, lengths):
        lengths, sorted_idx = lengths.sort(dim=0, descending=True)  # sorting ids
        _, unsorted_idx = sorted_idx.sort(dim=0)  # unsorting ids
        x_seqs = x_seqs[:, sorted_idx]  # sorting seqs by length
        x_packed = pack_padded_sequence(x_seqs, lengths.cpu())
        return x_packed, unsorted_idx

    @staticmethod
    def unpack_sequences(h_last_layer, h, unsorted_idx):
        h_last_layer_padded, h_last_layer_lengths = pad_packed_sequence(h_last_layer)
        h_last_layer = h_last_layer_padded[:, unsorted_idx]
        h = h[:, unsorted_idx]
        return h_last_layer, h

    def forward(self, y_seqs, hidden, lengths=None):
        """" Used for teacher forcing """
        y_seqs = self.emb(y_seqs.long())
        if self.packed_sequences and (lengths is not None):
            y_seqs, unsorted_idx = self.pack_sequences(y_seqs, lengths)  # Packing
            h_last_layer, h = self.gru(y_seqs, hidden)
            h_last_layer, h = self.unpack_sequences(h_last_layer, h, unsorted_idx)  # Unpacking
        else:
            h_last_layer, h = self.gru(y_seqs, hidden)

        preds = torch.stack([self.head(h) for h in h_last_layer])
        return preds, h

    def generate(self, hidden, lenseq=None, sos=2, eos=1, pad=0, force_length=True):
        """" Used for inference """
        if lenseq is None:
            lenseq = 100
        batch_size = hidden.shape[1]
        nb_eos = torch.zeros((1, batch_size), device=self.device)
        start = torch.tensor([sos] * batch_size, device=self.device).unsqueeze(0)
        seq_preds = []
        seq_inputs = [start]
        h = hidden
        while (len(seq_preds) < lenseq) and (nb_eos.min() == 0 or force_length):
            preds, h = self.forward(seq_inputs[-1], h, lengths=None)
            next_word = preds.argmax(dim=-1)
            nb_eos += (next_word == eos)
            # if the eos has already been encountered replace whatever
            # next_word is by pad id :
            seq_inputs.append(next_word.masked_fill_(nb_eos > 1, pad))
            seq_preds.append(preds)
        tensor_preds = torch.cat(seq_preds)
        return tensor_preds, h

        # start = torch.ones((1, batch_size)) * sos
        # start = start.to(self.device)
        # seq_preds = []
        # seq_inputs = [start]
        # h = hidden
        # # TODO replace for loop by while loop ?
        # for t in range(lenseq):
        #     preds, h = self.forward(seq_inputs[t], h, lengths=None)
        #     # TODO: sampling
        #     # TODO: Check EOS generation to stop predicted sequence using a mask
        #     seq_inputs.append(preds.argmax(dim=-1))
        #     seq_preds.append(preds)
        # tensor_preds = torch.cat(seq_preds)
        # return tensor_preds, h
