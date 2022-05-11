import torch
from torch import nn


class RNN(nn.Module):
    def __init__(self, input_dim, latent_dim, output_dim,
                 num_layers_enc, activation_enc, activation_dec,
                 device, num_layers_dec=1):
        super().__init__()
        self.name = "rnn"
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.latent_dim = latent_dim
        self.num_layers_enc = num_layers_enc
        self.num_layers_dec = num_layers_dec
        self.device = device
        self.activation_enc = activation_enc
        self.activation_dec = activation_dec

        self.linears_in = nn.ModuleList([nn.Linear(input_dim, latent_dim)])
        self.linears_h = nn.ModuleList([nn.Linear(latent_dim, latent_dim)])
        for i in range(self.num_layers_enc-1):
            self.linears_in.append(nn.Linear(latent_dim, latent_dim))
            self.linears_h.append(nn.Linear(latent_dim, latent_dim))

        self.decoder = nn.Sequential(nn.Linear(latent_dim, output_dim), self.activation_dec)

        self.to(device)

    def to(self, device):
        self.device = device
        super().to(device)

    def one_step(self, x, h_in):
        """
        one_step take an element x (generally from an input sequence) and unrolls it through all layers.
        h_in must provide an input hidden state for each rnn layer. Therefore h_init must be a tensor of shape
        (num_layers, batch_size, latent_dim) with latent_dim the latent dimension for the corresponding
        layer of the encoder. If h_init shape is

        outputs : hs a tensor of shape (num_layers, batch_size, latent_dim)
        """
        x = x.to(self.device)
        hs = [x]
        for i in range(self.num_layers_enc):
            h_i_1 = self.activation_enc(self.linears_in[i](hs[i]) + self.linears_h[i](h_in[i]))
            hs.append(h_i_1)
        return torch.stack(hs[1:])

    def forward(self, seq, h_init=None):
        """
        Take a sequence and apply one_step to each element of the sequence.
        Input :
            - seq a tensor of shape (seq_len, batch_size, input_dim)
        outputs :
            - hs_seq a tensor of shape (seq_len, num_layers, batch_size, latent_dim) containing all hidden states
            - h_last_layer a tensor of shape (seq_len, batch_size, latent_dim) containing the hidden states of the
            last layer
            - h_last a tensor of shape (num_layers, batch_size, latent_dim) containing the hidden states for each layer
             of the last element of the sequence
        """
        seq_len, batch_size, _ = seq.shape
        if h_init is None:
            h_init = torch.zeros((self.num_layers_enc, batch_size, self.latent_dim), device=self.device)

        h_seq = [h_init]
        for t in range(seq_len):
            h_t_1 = self.one_step(seq[t], h_seq[t])
            h_seq.append(h_t_1)

        h_seq_layers = torch.stack(h_seq[1:])  # dimensions : (seq_len, num_layers, batch_size, latent_dim)
        h_last_layer = h_seq_layers[:, -1, :, :]  # dimensions : (seq_len, batch_size, latent_dim)
        h_last = h_seq[-1]  # dimensions : (num_layers, batch_size, latent_dim)
        return h_last_layer, h_last

    def decode(self, h_seq):
        """
        Apply decoder on each hidden state of a given hidden state sequence.
        Typically used one the hidden state of the last layer.
        Input :
            - h_seq a tensor of shape (seq_len, batch_size, laten_dim) containing a sequence of hidden states
        Output :
            - out_seq a tensor of shape (seq_len, batch_size, output_dim) containing the decoded sequence.
        """
        assert(len(h_seq.shape) == 3)
        assert(h_seq.shape[2] == self.latent_dim)
        seq_len, batch_size, latent_dim = h_seq.shape
        return torch.stack([self.decoder(h_seq[t]) for t in range(seq_len)])


class MonoVariateRNN(nn.Module):
    def __init__(self, input_dim, latent_dim, num_layers_enc, activation_enc, activation_dec, device):
        super().__init__()
        self.n_models = input_dim
        self.models = nn.ModuleList([])
        for i in range(self.n_models):
            self.models.append(RNN(1, latent_dim, 1, num_layers_enc, activation_enc, activation_dec, device))

