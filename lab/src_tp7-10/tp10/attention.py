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


class SelfAttentionLayer(nn.Module):
    def __init__(self, input_dim, dropout=0.):
        super().__init__()
        self.input_dim = input_dim
        self.q_net = nn.Linear(self.input_dim, self.input_dim, bias=False)
        self.k_net = nn.Linear(self.input_dim, self.input_dim, bias=False)
        self.v_net = nn.Linear(self.input_dim, self.input_dim, bias=False)
        # self.in_proj = nn.Linear(self.input_dim, self.input_dim, bias=False)
        self.out_proj = nn.Linear(self.input_dim, self.input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask_pad):
        q = self.q_net(x)
        k = self.k_net(x)
        v = self.v_net(x)
        attn_output_weights = torch.bmm(q.transpose(1, 0), k.permute(1, 2, 0))
        attn_output_weights = attn_output_weights.masked_fill(
            mask_pad.transpose(0, 1).unsqueeze(1),
            float("-inf"))

        attn_output_weights = attn_output_weights / (k.shape[-1] ** (1/4))

        attn_output_weights = F.softmax(attn_output_weights, dim=-1)
        attn_output_weights = self.dropout(attn_output_weights)

        attn_output = torch.bmm(attn_output_weights, v.transpose(1, 0)).transpose(1, 0)
        attn_output = self.out_proj(attn_output)
        return attn_output, attn_output_weights


class TransformerEncoderLayer(nn.Module):
    def __init__(self, input_dim, dropout=0.1):
        super().__init__()

        self.input_dim = input_dim
        self.self_attention = SelfAttentionLayer(self.input_dim, dropout=dropout)
        self.mlp = nn.Sequential(nn.Linear(input_dim, 2 * input_dim),
                                 nn.ReLU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(2 * input_dim, input_dim))

        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask_pad):
        x_att, _ = self.self_attention(x, mask_pad)
        x = x + self.dropout1(x_att)
        x = self.norm1(x)
        z = self.mlp(x)
        x = x + self.dropout2(z)
        x = self.norm2(x)
        return x


class TransformerModel(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers, cls=False):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.cls = cls
        self.transformers = nn.ModuleList()
        self.classifier = nn.Linear(input_dim, output_dim, bias=True)

        for i in range(self.num_layers):
            self.transformers.append(TransformerEncoderLayer(input_dim))

    def forward(self, x, mask_pad):
        out = x
        for i in range(self.num_layers):
            out = self.transformers[i](out, mask_pad)
        if self.cls:
            out_global = out[0]
        else:
            out_global = compute_masked_mean(out, mask_pad)
        return self.classifier(out_global)


# class TransformerModel(nn.Module):
#     def __init__(self, input_dim, output_dim, num_layers):
#         super().__init__()
#
#         self.input_dim = input_dim
#         self.output_dim = output_dim
#         self.transformer = nn.TransformerEncoderLayer(input_dim, nhead=1, dim_feedforward=input_dim)
#         self.classifier = nn.Linear(self.input_dim, self.output_dim, bias=True)
#
#     def forward(self, x, mask_pad):
#         out = self.transformer(x, src_key_padding_mask=mask_pad.transpose(1, 0))
#         out_mean = compute_masked_mean(out, mask_pad)
#         out = self.classifier(out_mean)
#         return out