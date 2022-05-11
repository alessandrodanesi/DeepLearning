import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


def compute_masked_mean(x, mask):
    mask = mask.unsqueeze(-1)  # (x_len,bsize,1)
    not_mask = ~mask
    x_len = not_mask.sum(dim=0)
    mean_x = ((not_mask * x).sum(dim=0) / x_len)
    return mean_x


# #### Q1 #### #
class AttentionLayerQ1(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim

    @staticmethod
    def forward(x, mask_pad):
        attn = compute_masked_mean(x, mask_pad)
        return attn, attn


class AttentionModelQ1(nn.Module):
    def __init__(self, input_dim, out_dim):
        super().__init__()

        self.input_dim = input_dim
        self.out_dim = out_dim
        self.attention_layer = AttentionLayerQ1(self.input_dim)
        self.classifier = nn.Linear(self.input_dim, out_dim, bias=True)

    def forward(self, x, mask_pad):
        z, attn = self.attention_layer(x, mask_pad)
        pred = self.classifier(z)
        return pred, attn


# #### Q2 #### #
class AttentionLayerQ2(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        self.q = nn.Parameter(torch.randn((self.input_dim, 1), requires_grad=True))
        self.cste = nn.Parameter(torch.randn((1,), requires_grad=True))
        self.k_net = nn.Identity()
        self.v_net = nn.Identity()

    def compute_masked_attention_weights(self, q, k, mask_pad):
        # q : # (bsize, in_dim), k : (x_len, bsize, in_dim)
        attn = k @ q  # (x_len, bsize, 1)
        attn = (self.cste + attn) / np.sqrt(k.shape[-1])
        attn[mask_pad] = float("-Inf")
        attn = F.softmax(attn, dim=0)  # (x_len,bsize,1)
        return attn

    def forward(self, x, mask_pad):
        k = self.k_net(x)
        v = self.v_net(x)
        attn = self.compute_masked_attention_weights(self.q, k, mask_pad)
        out = torch.bmm(attn.permute(1, 2, 0), v.transpose(1, 0))
        out = out.transpose(1, 0)
        return out, attn


class AttentionModelQ2(nn.Module):
    def __init__(self, input_dim, out_dim):
        super().__init__()

        self.input_dim = input_dim
        self.out_dim = out_dim
        self.attention_layer = AttentionLayerQ2(self.input_dim)
        self.classifier = nn.Linear(self.input_dim, out_dim, bias=True)

    def forward(self, x, mask_pad):
        z, attn = self.attention_layer(x, mask_pad)
        pred = self.classifier(z)
        return pred, attn


# #### Q3 #### #
class AttentionLayer(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        self.q_net = nn.Sequential(nn.Linear(self.input_dim, self.input_dim, bias=False))
        self.k_net = nn.Sequential(nn.Linear(self.input_dim, self.input_dim, bias=False))
        self.v_net = nn.Sequential(nn.Linear(self.input_dim, self.input_dim, bias=False))
        self.cste = nn.Parameter(torch.randn((1,), requires_grad=True))

    def compute_masked_attention_weights(self, q, k, mask_pad):
        # q : # (bsize, in_dim), k : (x_len, bsize, in_dim)
        attn = torch.bmm(q.unsqueeze(1), k.permute(1, 2, 0).contiguous())  # (bsize, 1, x_len)
        attn = attn.permute(2, 0, 1)  # (x_len,bsize,1)
        attn = (self.cste + attn) / np.sqrt(k.shape[-1])
        attn[mask_pad] = float("-Inf")
        attn = F.softmax(attn, dim=0)  # (x_len, bsize, 1)
        return attn

    def forward(self, x, mask_pad):
        mean_x = compute_masked_mean(x, mask_pad)
        q = self.q_net(mean_x)
        k = self.k_net(x)
        v = self.v_net(x)
        attn = self.compute_masked_attention_weights(q, k, mask_pad)  # (x_len, bsize, 1)
        # (bsize, 1, x_len) * (bsize, x_len, in_dim) -> (b_size, 1, in_dim)
        out = torch.bmm(attn.permute(1, 2, 0), v.transpose(1, 0))
        out = out.transpose(1, 0)
        return out, attn


class AttentionModelQ3(nn.Module):
    def __init__(self, input_dim, out_dim):
        super().__init__()

        self.input_dim = input_dim
        self.out_dim = out_dim
        self.attention_layer = AttentionLayer(self.input_dim)
        self.classifier = nn.Linear(self.input_dim, out_dim, bias=True)

    def forward(self, x, mask_pad):
        z, attn = self.attention_layer(x, mask_pad)
        pred = self.classifier(z)
        return pred, attn


# #### Q4 #### #
class AttentionModelQ4(nn.Module):
    def __init__(self, input_dim, out_dim, num_rec_layers):
        super().__init__()

        self.input_dim = input_dim
        self.out_dim = out_dim
        self.gru = nn.GRU(self.input_dim, self.input_dim, num_rec_layers)
        self.attention_layer = AttentionLayer(self.input_dim)
        self.classifier = nn.Linear(self.input_dim, out_dim, bias=True)

    def forward(self, x, mask_pad):
        h_last_layer, h_end = self.gru(x)
        z, attn = self.attention_layer(h_last_layer, mask_pad)
        pred = self.classifier(z)
        return pred, attn