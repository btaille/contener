import torch
import torch.nn as nn

from utils.torch_utils import mask

class WordEmbedder(nn.Module):
    def __init__(self, word_vocab, embed_dim, w_embeddings=None, freeze=False):
        super(WordEmbedder, self).__init__()

        self.word2idx, _ = word_vocab
        self.w_embed_dim = embed_dim
        self.freeze_embeddings = freeze
        self.name = "word_embedder"

        # layers
        if w_embeddings is None:
            w_embeddings = 1e-3 * torch.randn(len(self.word2idx), self.w_embed_dim)
            assert not self.freeze_embeddings
        else:
            assert embed_dim == w_embeddings.shape[1]

        self.word_embeddings = nn.Embedding(w_embeddings.size(0), w_embeddings.size(1))
        self.word_embeddings.weight = nn.Parameter(w_embeddings)

        if self.freeze_embeddings:
            self.word_embeddings.weight.requires_grad = False

    def forward(self, data):
        # Word embeddings
        embeds = self.word_embeddings(data["words"])
        return {"embeddings": embeds}
