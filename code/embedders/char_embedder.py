import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.pooling import MaxPool2d, AvgPool2d

from utils.rnn_utils import PackedRNN


class CharBiLSTMPool(nn.Module):
    def __init__(self, char_vocab, char_embed_dim=100, char_hidden_dim=25, pool="last", dropout=0.):
        super(CharBiLSTMPool, self).__init__()
        assert pool in ["max", "avg", "last"], "'{}' should be max, avg or last.".format(pool)

        self.char2idx, _ = char_vocab
        self.char_hidden_dim = char_hidden_dim
        self.char_embed_dim = char_embed_dim
        self.w_embed_dim = 2 * char_hidden_dim

        self.pool = pool

        self.drop = nn.Dropout(dropout)

        if self.pool == "max":
            self.pool_fn = MaxPool2d
        elif self.pool == "avg":
            self.pool_fn = AvgPool2d

        char_embeddings = 1e-3 * torch.randn(len(self.char2idx), self.char_embed_dim)
        self.char_embeddings = nn.Embedding(char_embeddings.size(0), char_embeddings.size(1))
        self.char_embeddings.weight = nn.Parameter(char_embeddings)

        char_lstm = nn.LSTM(char_embeddings.size(1), self.char_hidden_dim, bidirectional=True, batch_first=True)
        self.packed_char_lstm = PackedRNN(char_lstm)
        self.name = "char"

    def forward(self, data):
        char_embeddings = self.drop(self.char_embeddings(data["chars"]))
        batch_size, max_nwords, max_nchars, char_embedding_dim = char_embeddings.size()

        # Flatten for LSTM
        # [batch, max words, max chars, char embedding] -> [batch * max words, max chars, char embedding]
        flat_embeddings = char_embeddings.view(-1, max_nchars, char_embedding_dim)
        flat_lens = np.concatenate(data["nchars"])

        # LSTM
        flat_out, char_hidden = self.packed_char_lstm(flat_embeddings, flat_lens)

        if self.pool in ["max", "avg"]:
            # Pooling 
            pooled_flat_out = self.pool_fn([flat_out.size(1), 1])(flat_out).view(-1, 2 * self.char_hidden_dim)
        elif self.pool == "last":
            # Unpack and pad last hidden states
            pooled_flat_out = torch.cat((char_hidden[0][0], char_hidden[0][1]), -1)

        # Unflatten
        return {"embeddings": pooled_flat_out.view(batch_size, -1, 2 * self.char_hidden_dim)}
