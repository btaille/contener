import torch
from torch import nn

from global_vars import EMBEDDINGS_DIR
from embedders.bert_embedder import BertEmbedder
from embedders.char_embedder import CharBiLSTMPool
from embedders.elmo_embedder import ElmoEmbedder, ElmoLayerEmbedder
from embedders.flair_embedder import FlairEmbedder
from embedders.word_embedder import WordEmbedder


class Embedder(nn.Module):
    def __init__(self, mode, w_embeddings=None, params=None, word_vocab=None, char_vocab=None,
                 elmo_dropout=0.,
                 flair_word=None, device="cuda", num_output_representations=1):
        super(Embedder, self).__init__()
        if mode == "elmo":
            options_file = EMBEDDINGS_DIR + "elmo/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json"
            weight_file = EMBEDDINGS_DIR + "elmo/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5"
            self.embedder = ElmoEmbedder(options_file, weight_file, num_output_representations, dropout=elmo_dropout,
                                         device=device)

        elif mode == "elmo_zero":
            options_file = EMBEDDINGS_DIR + "elmo/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json"
            weight_file = EMBEDDINGS_DIR + "elmo/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5"
            self.embedder = ElmoLayerEmbedder(0, options_file, weight_file, 1, dropout=elmo_dropout, device=device)

        elif mode == "glove":
            assert w_embeddings is not None
            assert params is not None
            assert word_vocab is not None

            self.embedder = WordEmbedder(word_vocab, params["glove_word_embedding_dim"],
                                         w_embeddings=w_embeddings,
                                         freeze=params["freeze_embeddings"])

        elif mode == "random":
            assert params is not None
            assert word_vocab is not None
            self.embedder = WordEmbedder(word_vocab["train"], params["word_embedding_dim"], "train",
                                         freeze=False)

        elif mode == "flair":
            self.embedder = FlairEmbedder("news", word=flair_word)
        elif mode == "bert-base":
            self.embedder = BertEmbedder("bert-base-cased", finetune=True, device=device)
        elif mode == "bert-large":
            self.embedder = BertEmbedder("bert-large-cased", finetune=True, device=device)
        elif mode == "bert-base-fb":
            self.embedder = BertEmbedder("bert-base-cased", finetune=False, device=device)
        elif mode == "bert-large-fb":
            self.embedder = BertEmbedder("bert-large-cased", finetune=False, device=device)
        elif mode == "char":
            assert char_vocab is not None
            assert params is not None
            self.embedder = CharBiLSTMPool(char_vocab,
                                           dropout=params["dropout"],
                                           char_embed_dim=params["char_embedding_dim"],
                                           char_hidden_dim=params["char_hidden_dim"],
                                           pool=params["char_pool"])
        else:
            print(mode)
            assert False

        if mode in ["bert-base-fb", "bert-large-fb"]:
            for param in self.embedder.parameters():
                param.requires_grad = False

        self.name = self.embedder.name
        self.w_embed_dim = self.embedder.w_embed_dim


    def forward(self, data):
        return self.embedder(data)


class StackedEmbedder(nn.Module):
    def __init__(self, embedders, proj_dim=0):
        super(StackedEmbedder, self).__init__()
        self.embedders = {embedder.name: embedder for embedder in embedders}

        self.w_embed_dim = 0

        for name, embedder in self.embedders.items():
            self.add_module(name, embedder)
            self.w_embed_dim += embedder.w_embed_dim

        self.projector = None
        if proj_dim and not proj_dim == self.w_embed_dim:
            self.projector = nn.Linear(self.w_embed_dim, proj_dim)
            self.w_embed_dim = proj_dim

    def forward(self, data):
        if len(self.embedders) == 1:
            for name, embedder in self.embedders.items():
                return embedder(data)

        else:
            embeddings = []
            for name, embedder in self.embedders.items():
                embeddings.append(embedder(data)["embeddings"])

            embeddings = torch.cat(embeddings, -1)
            return {"embeddings": embeddings}
