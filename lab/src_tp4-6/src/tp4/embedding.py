import torch


class OneHotEmbedding:
    def __init__(self, embedding_size, device):
        self.embedding_size = embedding_size
        self.device = device

    def to(self, device):
        self.device = device

    def __call__(self, tensor_indices):
        """
        Takes an input tensor of batch of sequences of caracter indices (seq_len, batch_size) an return a batch of
        sequences of onehot vector (seq_len, batch_size, embedding_size)
        """
        len_seq, batch_size = tensor_indices.shape
        one_hot = torch.zeros(len_seq, batch_size, self.embedding_size, device=self.device)
        one_hot.scatter_(2, tensor_indices.unsqueeze(2), 1)
        return one_hot

