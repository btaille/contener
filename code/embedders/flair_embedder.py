import torch
from flair.data import Sentence
from flair.embeddings import FlairEmbeddings, WordEmbeddings, StackedEmbeddings
from torch import nn


def detokenize(sent):
    return " ".join([w for w in sent if not w == '<PAD>'])


def pad(tensors, padlen=None):
    if padlen is None:
        padlen = max([len(t) for t in tensors])

    padded_batch = torch.zeros((len(tensors), padlen, tensors[0].size(-1)))
    for i, t in enumerate(tensors):
        padded_batch[i, :len(t), :] = t

    return padded_batch


class FlairEmbedder(nn.Module):
    def __init__(self, flair_model="news", word="glove", finetune=False):
        super(FlairEmbedder, self).__init__()

        self.flair_model = flair_model
        self.word = word
        self.finetune = finetune

        embeddings = [FlairEmbeddings('{}-forward'.format(self.flair_model)),
                      FlairEmbeddings('{}-backward'.format(self.flair_model))]

        if self.word is not None:
            embeddings.append(WordEmbeddings(self.word))

        self.stacked_embeddings = StackedEmbeddings(embeddings)
        self.w_embed_dim = self.stacked_embeddings.embedding_length
        self.name = "flair"

    def forward(self, data):
        device = data["words"].device
        sentences = [Sentence(detokenize(s)) for s in data["sents"]]

        if self.finetune:
            self.stacked_embeddings.embed(sentences)

        else:
            self.stacked_embeddings.eval()
            with torch.no_grad():
                self.stacked_embeddings.embed(sentences)

        tensors = [torch.cat([token.embedding.unsqueeze(0) for token in sent], dim=0) for sent in sentences]

        return {"embeddings": pad(tensors).to(device)}
