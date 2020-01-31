import numpy as np
import torch


def parse_sent(sent):
    words = sent.split()
    nwords = len(words)
    chars = [[c for c in w] for w in words]
    nchars = [len(w) for w in chars]

    data = {"words": words,
            "chars": chars,
            "nwords": nwords,
            "nchars": nchars}

    return data


def parse(data, vocab):
    parsed = {}

    sents = data["words"]
    words = [sent.split() for sent in sents]
    chars = [[[c for c in w] for w in sent] for sent in words]
    nwords = [len(s) for s in words]
    nchars = [[len(w) for w in sent] for sent in chars]

    w2idx, idx2w = vocab["word"]
    c2idx, idx2c = vocab["char"]
    t2idx, idx2t = vocab["tag"]

    word_ids = np.array(seq2ids(words, w2idx))
    char_ids = chars2ids(chars, c2idx)

    update = {"words": word_ids,
              "chars": np.array(char_ids),
              "sents": np.array(words),
              "nwords": np.array(nwords),
              "nchars": np.array(nchars)}
    parsed.update(update)

    if "tags" in data.keys():
        tags = [s.split() for s in data["tags"]]
        tags = seq2ids(tags, t2idx)
        parsed.update({"tags": np.array(tags)})

    if "relations" in data.keys():
        parsed.update({"relations": np.array(data["relations"])})

    return parsed


def seq2ids(sequences, w2idx, oov=1, cased=False):
    if cased:
        return [[w2idx[w] if w in w2idx.keys() else oov for w in seq] for seq in sequences]
    else:
        output = []
        for seq in sequences:
            out_seq = []
            for w in seq:
                if w in w2idx.keys():
                    out_seq.append(w2idx[w])
                elif w.lower() in w2idx.keys():
                    out_seq.append(w2idx[w.lower()])
                else:
                    out_seq.append(oov)
            output.append(out_seq)
        return output


def chars2ids(chars, c2idx, oov=1):
    return [[[c2idx[c] if c in c2idx.keys() else oov for c in w] for w in s] for s in chars]


def ids2seq(idxs, idx2w):
    return [idx2w[idx] for idx in idxs]


def pad(sequences, pad=0):
    maxlen = max([len(seq) for seq in sequences])
    return [seq + [pad] * (maxlen - len(seq)) for seq in sequences]


def pad_batch1D(sequences, pad_shape, pad=0, device="cpu", dtype=torch.long):
    padded = pad * torch.ones(pad_shape).to(dtype)
    for i, s in enumerate(sequences):
        padded[i][:len(s)] = torch.tensor(s)
    return padded.to(device)


def pad_batch2D(sequences, pad_shape, pad=0, device="cpu", dtype=torch.long):
    padded = pad * torch.ones(pad_shape).to(dtype)
    for i, s in enumerate(sequences):
        for j, w in enumerate(s):
            padded[i, j][:len(w)] = torch.tensor(w)
    return padded.to(device)


def onehot(idxs, ntags):
    onehots = np.zeros((len(idxs), ntags), dtype=np.int32)
    for k, idx in enumerate(idxs):
        if idx < ntags:
            onehots[k][idx] = 1
    return onehots


def decreasing_sort(data):
    words = data["words"]
    nwords = data["nwords"]
    chars = data["chars"]
    nchars = data["nchars"]

    # Negative stride not supported in pytorch
    indices = np.array(np.argsort(nwords)[::-1])

    words = words[indices]
    nwords = nwords[indices]
    chars = chars[indices]
    nchars = nchars[indices]

    data.update({"words": np.array(words),
                 "chars": np.array(chars),
                 "nwords": np.array(nwords),
                 "nchars": np.array(nchars)})

    if "tags" in data.keys():
        tags = data["tags"]
        data.update({"tags": tags[indices]})

    return data, indices


class DataIterator(object):
    def __init__(self, data, dicts, device="cpu", batch=1, drop_last=False, shuffle=False):
        self.word_dicts = dicts["word"]
        self.c2idx, self.idx2c = dicts["char"]
        self.t2idx, self.idx2t = dicts["tag"]

        if "relations" in dicts.keys():
            self.r2idx, self.idx2r = dicts["relations"]

        self.data = parse(data, dicts)

        self.shuffle = shuffle

        if "tags" not in self.data.keys():
            self.shuffle = False

        self.batch = batch

        self.device = device

        self.i = 0
        self.max = len(self.data["sents"])

        if drop_last:
            self.max -= self.max % self.batch

        if self.shuffle:
            self.shuffle_data()

        self.nbatches = self.max // self.batch + int(self.max % self.batch > 0)

    def __iter__(self):
        return self

    def __next__(self):
        if self.i < self.max:
            ranged = (self.i, min(self.i + self.batch, self.max))
            self.i += self.batch

            batch = {}
            batch["nwords"] = self.data["nwords"][ranged[0]:ranged[1]]
            batch["nchars"] = self.data["nchars"][ranged[0]:ranged[1]]

            batch["sents"] = self.data["sents"][ranged[0]:ranged[1]]

            batch_len = len(batch["nwords"])
            word_padlen = max(batch["nwords"])
            char_padlen = max([max([nc for nc in s]) for s in batch["nchars"]])

            batch["nchars"] = [nc + [1] * (word_padlen - len(nc)) for nc in batch["nchars"]]

            batch["words"] = pad_batch1D(self.data["words"][ranged[0]:ranged[1]], (batch_len, word_padlen), pad=0,
                                         dtype=torch.long, device=self.device)
            batch["chars"] = pad_batch2D(self.data["chars"][ranged[0]:ranged[1]], (batch_len, word_padlen, char_padlen),
                                         pad=0, dtype=torch.long, device=self.device)

            if "tags" in self.data.keys():
                batch["tags"] = pad_batch1D(self.data["tags"][ranged[0]:ranged[1]], (batch_len, word_padlen),
                                            pad=self.t2idx["O"],
                                            dtype=torch.long, device=self.device)

            return batch
        raise StopIteration

    def reinit(self):
        self.i = 0
        if self.shuffle:
            self.shuffle_data()

    def shuffle_data(self):
        indices = np.random.permutation(np.arange(self.max))

        self.data["words"] = self.data["words"][indices]
        self.data["chars"] = self.data["chars"][indices]
        self.data["nwords"], self.data["nchars"] = self.data["nwords"][indices], \
                                                   self.data["nchars"][indices]
        self.data["sents"] = self.data["sents"][indices]

        if "tags" in self.data.keys():
            self.data["tags"] = self.data["tags"][indices]


def reorder(sorted_arr, argsort):
    return sorted_arr[np.argsort(argsort)]


def merge(to_merge):
    if type(to_merge[0]) == list:
        return np.concatenate(to_merge)

    elif type(to_merge[0]) == dict:
        keys = to_merge[0].keys()
        merged = {k: merge([elt[k] for i, elt in enumerate(to_merge)]) for k in keys}
        return merged


def init_iterators(data, dicts, batch_size=128, device="cuda", shuffle_train=True, train_dev=False):
    train_iterator = DataIterator(data["train"], dicts, batch=batch_size, device=device, shuffle=shuffle_train)
    dev_iterator = DataIterator(data["dev"], dicts, batch=batch_size, device=device)
    test_iterator = DataIterator(data["test"], dicts, batch=batch_size, device=device)

    iterators = {"train": train_iterator, "dev": dev_iterator, "test": test_iterator}

    if train_dev:
        iterators.update({"train+dev": DataIterator(merge([data["train"], data["dev"]]), dicts, batch=batch_size,
                                                    device=device, shuffle=shuffle_train)})

    return iterators
