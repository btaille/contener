import torch
from allennlp.modules.conditional_random_field import ConditionalRandomField as CRF
from torch import nn

from data.data_iterator import pad_batch1D, ids2seq
from utils.torch_utils import mask


class NERDecoder(nn.Module):
    def __init__(self, hidden_dim, tag_vocab, dropout=0., name="ner"):
        super(NERDecoder, self).__init__()
        # parameters
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.tag2idx, self.idx2tag = tag_vocab

        self.num_tags = len(self.tag2idx)

        # layers
        self.drop = nn.Dropout(self.dropout)

        self.name = name
        self.supervision = "tags"

    def decode(self, predictions, seqlens):
        preds = []
        for i, p in enumerate(predictions):
            preds.append(ids2seq(predictions[i, :seqlens[i]].tolist(), self.idx2tag))
        return preds


class SoftmaxDecoder(NERDecoder):
    def __init__(self, hidden_dim, tag_vocab, dropout=0.):
        super(SoftmaxDecoder, self).__init__(hidden_dim, tag_vocab, dropout=dropout)
        self.hidden2tag = nn.Linear(self.hidden_dim, self.num_tags)

    def forward(self, data, input_key="encoded", detach=False):
        x = data[input_key]

        if detach:
            x = x.detach()

        encoded = self.drop(x)

        scores = self.hidden2tag(encoded)
        scores = nn.Softmax(dim=-1)(scores)
        data["{}_output".format(self.name)] = torch.argmax(scores, dim=-1)
        data["{}_scores".format(self.name)] = scores
        return data["{}_output".format(self.name)], data["{}_scores".format(self.name)]

    def loss(self, data, normalization="batch"):
        scores = data["{}_scores".format(self.name)]
        tags = data[self.supervision]
        seqlens = data["nwords"]

        loss = nn.CrossEntropyLoss(reduction="none")(scores.view(-1, len(self.idx2tag)), tags.view(-1))
        masked_loss = loss * mask(seqlens).to(scores.device).view(-1).float()

        if normalization == "words":
            return masked_loss.sum() / mask(seqlens).to(scores.device).sum().float()
        elif normalization == "batch":
            return masked_loss.sum() / len(masked_loss)


class CRFDecoder(NERDecoder):
    def __init__(self, hidden_dim, tag_vocab, dropout=0., normalization="none"):
        super(CRFDecoder, self).__init__(hidden_dim, tag_vocab, dropout=dropout)
        self.hidden2tag = nn.Linear(self.hidden_dim, self.num_tags)
        self.crf = CRF(self.num_tags)
        self.normalization = normalization

    def forward(self, data, input_key="encoded", detach=False):
        x = data[input_key]

        if detach:
            x = x.detach()

        encoded = self.drop(x)
        feats = self.hidden2tag(encoded)

        tags, _ = zip(*self.crf.viterbi_tags(feats, mask(data["nwords"])))
        padded_tags = pad_batch1D(tags, feats.size()[:2], pad=0, dtype=torch.float, device=encoded.device)

        data["{}_output".format(self.name)] = padded_tags
        data["{}_scores".format(self.name)] = feats
        return data["{}_output".format(self.name)], data["{}_scores".format(self.name)]

    def loss(self, data):
        feats = data["{}_scores".format(self.name)]
        tags = data[self.supervision]
        seqlens = data["nwords"]

        if self.normalization == "batch":
            return - self.crf(feats, tags, mask=mask(seqlens).to(feats.device)) / feats.size(0)
        else:
            return - self.crf(feats, tags, mask=mask(seqlens).to(feats.device))