import torch
from torch.nn import functional as F
from text_utils import string2code, code2string
from torch.distributions import Categorical
import string
import random


def generate(rnn, emb, eos, start="", maxlen=200, stochastic=False):
    """
    Génération à partir du RNN
    """
    if start == "":
        start = random.sample(string.ascii_uppercase, 1)[0]
    sentence = start
    seq = string2code(start).to(rnn.device)
    pred_car = None
    while len(seq) < maxlen and pred_car != eos:
        seq_emb = emb(seq.long().unsqueeze(1))  # to make it batch_size = 1
        h_last_layer, h_last = rnn(seq_emb)
        output = rnn.decoder(h_last[-1])  # -1 because h_last is dim (num_layers, batch_size = 1, latent_dim)
        logits = F.log_softmax(output, dim=-1)
        if stochastic:
            m = Categorical(logits=logits)
            pred_ix = m.sample().flatten()
        else:
            pred_ix = logits.argmax(dim=-1).flatten()
        pred_car = code2string(pred_ix)
        sentence += pred_car
        seq = torch.cat([seq, pred_ix])
    return sentence


def forecast(rnn, seq, n_steps):
    """
    Performs n_steps forcasting for a given sequence. Only usable if input_dim == output_dim.
    Input :
        - seq a tensor of shape (seq_len, batch_size, input_dim) containing the input sequence
    Output :
        - y_preds a tensor of shape (n_steps, batch_size, output_dim) with output_dim == input_dim.
    """
    seq = seq.to(rnn.device)
    assert(rnn.input_dim == rnn.output_dim)
    h_last_layer, h_last = rnn(seq)
    h_preds = [h_last]
    y_preds = []
    for t in range(n_steps):
        y_preds.append(rnn.decoder(h_preds[t][-1]))  # use only the encoding of the last layer to decode
        h_preds.append(rnn.one_step(y_preds[t], h_preds[t]))  # the last one is not used
    return torch.stack(y_preds)
