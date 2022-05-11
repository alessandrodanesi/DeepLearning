from textloader import code2string, string2code
import torch
from torch.nn import functional as F
from torch.distributions import Categorical
from itertools import chain
from operator import itemgetter
import string
import random
import numpy as np


def generate(rnn, emb, decoder, eos, start="", maxlen=100, stochastic=False):
    """
    Génération à partir du RNN, et d'une fonction decoder qui renvoie les logits (logarithme de probabilité à une
     constante près, i.e. ce qui vient avant le softmax) des différentes sorties possibles.
    """
    if start == "":
        start = random.sample(string.ascii_uppercase, 1)[0]

    sentence = start
    seq = string2code(start).to(rnn.device)
    pred_car = None
    while len(seq) < maxlen and pred_car != eos:
        seq_emb = emb(seq.long()).unsqueeze(1)  # to make it batch_size = 1
        h_last_layer, h_last = rnn(seq_emb)
        logits = decoder(h_last[-1])  # -1 because h_last is dim (num_layers, batch_size = 1, latent_dim)
        if stochastic:
            m = Categorical(logits=logits)
            pred_ix = m.sample().flatten()
        else:
            pred_ix = logits.argmax(dim=-1).flatten()
        pred_car = code2string(pred_ix)
        sentence += pred_car
        seq = torch.cat([seq, pred_ix])
    return sentence


def generate_beam(rnn, emb, decoder, eos, k, start="", maxlen=100):
    if start == "":
        start = random.sample(string.ascii_uppercase, 1)[0]
    seq = string2code(start).to(rnn.device)
    buffer = [(seq, 0)]
    eos_code = string2code(eos)
    depth = 0
    n_eos = int(seq[-1].item() == eos_code.item())
    while n_eos < len(buffer) and depth < maxlen:
        for i, (phrase, log_likelihood) in enumerate(buffer):
            if phrase[-1].item() == eos_code.item():
                buffer[i] = [(phrase, log_likelihood)]  # it is its own and only child
                continue
            seq_emb = emb(phrase.long()).unsqueeze(1)  # make it batch_size = 1
            h_last_layer, h_last = rnn(seq_emb)
            output = decoder(h_last[-1])
            logits = F.log_softmax(output, dim=-1)
            top_log_prob, top_codes = logits.sort(descending=True)
            top_k_log_prob, top_k_codes = top_log_prob[:, :k], top_codes[:, :k]
            # Expand parent to children
            children = [(torch.cat([phrase, top_k_codes[:, j]]), log_likelihood + top_k_log_prob[:, j].item()) for j in range(k)]
            # Replace parent by children
            buffer[i] = children
        # Flatten
        buffer = list(chain(*buffer))
        # Top K
        buffer = sorted(buffer, key=itemgetter(1))[-k:]
        depth = max(map(lambda x: x[0].shape[0], buffer))
        n_eos = sum([int(phrase[-1].item() == eos_code.item()) for phrase, _ in buffer])

    most_probable_sentence = buffer[-1][0]
    return code2string(most_probable_sentence)


def generate_nucleus(rnn, emb, decoder, eos, p=0.9, k=5, mode="pnucleus", start="", maxlen=100):
    if start == "":
        start = random.sample(string.ascii_uppercase, 1)[0]

    sentence = start
    seq = string2code(start).to(rnn.device)
    pred_car = None
    if mode == "pnucleus":
        compute_probs = p_nucleus(decoder, p)
    elif mode == "topk":
        compute_probs = top_k_sampling(decoder, k)

    while len(seq) < maxlen and pred_car != eos:
        seq_emb = emb(seq.long()).unsqueeze(1)  # to make it batch_size = 1
        h_last_layer, h_last = rnn(seq_emb)
        assert(isinstance(k, int))
        probs = compute_probs(h_last[-1])
        m = Categorical(probs=probs)
        pred_ix = m.sample().flatten()
        pred_car = code2string(pred_ix)
        sentence += pred_car
        seq = torch.cat([seq, pred_ix])
    return sentence


def top_k_sampling(decoder, k: int):
    """Renvoie une fonction qui calcule la distribution de probabilité sur les sorties

    Args:
        decoder: renvoie les logits étant donné l'état du RNN
        k (int): [description]
    """
    def compute(h):
        """Calcule la distribution de probabilité sur les sorties

        Args:
            h (torch.Tensor): L'état à décoder
        """
        output = decoder(h)
        probs = F.softmax(output, dim=-1)
        top_k_probs, top_k_codes = torch.topk(probs, k)
        probs_nucleus = torch.zeros_like(probs).scatter_(1, top_k_codes, top_k_probs)
        probs_nucleus = probs_nucleus / probs_nucleus.sum(dim=1, keepdim=True)
        return probs_nucleus
    return compute


def p_nucleus(decoder, p: float):
    """Renvoie une fonction qui calcule la distribution de probabilité sur les sorties

    Args:
        decoder: renvoie les logits étant donné l'état du RNN
        p (float): [description]
    """
    def compute(h):
        """Calcule la distribution de probabilité sur les sorties

        Args:
            h (torch.Tensor): L'état à décoder
        """
        output = decoder(h)
        probs = F.softmax(output, dim=-1)
        sorted_probs, sorted_codes = torch.sort(probs, descending=True)
        cumulative_probs = sorted_probs.cumsum(dim=-1)
        idx = torch.where(cumulative_probs <= p)[1]
        if idx.shape == torch.Size([0]):
            idx = torch.tensor([0], dtype=torch.int64)
        idx = torch.cat([idx, idx[-1].unsqueeze(0)+1])
        top_codes = sorted_codes[:, idx]
        top_probs = sorted_probs[:, idx]
        probs_nucleus = torch.zeros_like(probs).scatter_(1, top_codes, top_probs)
        probs_nucleus = probs_nucleus / probs_nucleus.sum(dim=1, keepdim=True)
        return probs_nucleus
    return compute

