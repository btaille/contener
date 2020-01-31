import os

import torch

from data.data_utils import load_file
from utils.embeddings import load_vocabulary, load_embeddings


def load_data(data_path,
              filenames=["train", "dev", "test"],
              splits=["train", "dev", "test"],
              embedding_path="glove.840B.300d",
              scheme="iobes",
              start_stop_tags=False):
    output = {}

    ### Load dataset
    data = {split: {} for split in splits}

    for split, fn in zip(splits, filenames):
        data[split]["words"] = [" ".join(s) for s in load_file(os.path.join(data_path, "{}.words.txt".format(fn)))]
        data[split]["tags"] = [" ".join(s) for s in load_file(os.path.join(data_path, "{}.{}.txt".format(fn, scheme)))]

    output.update({"data": data})

    ### Load vocabulary and embeddings
    word2idx, idx2word = load_vocabulary(os.path.join(data_path, "vocab.words.txt"), oov_padding=True)
    char2idx, idx2char = load_vocabulary(os.path.join(data_path, "vocab.chars.txt"), oov_padding=True)
    ner2idx, idx2ner = load_vocabulary(os.path.join(data_path, "vocab.iobes.txt"), start_stop=start_stop_tags)

    vocab = {"word": (word2idx, idx2word),
             "char": (char2idx, idx2char),
             "tag": (ner2idx, idx2ner)}

    output.update({"vocab": vocab})

    if embedding_path is not None and os.path.exists(os.path.join(data_path, embedding_path + "_embeddings.npz")):
        _, embeddings = load_embeddings(os.path.join(data_path, embedding_path))
        output.update({"embeddings": torch.Tensor(embeddings)})

    elif embedding_path is not None and os.path.exists(embedding_path + "_embeddings.npz"):
        _, embeddings = load_embeddings(os.path.join(data_path, embedding_path))
        output.update({"embeddings": torch.Tensor(embeddings)})

    else:
        output.update({"embeddings": None})

    ### Load overlap files
    if os.path.exists(os.path.join(data_path, "dev.overlap.txt")):
        dev_overlap = load_file(os.path.join(data_path, "dev.overlap.txt"))
        test_overlap = load_file(os.path.join(data_path, "test.overlap.txt"))

    else:
        dev_overlap = None
        test_overlap = None

    output.update({"overlap": {"dev": dev_overlap, "test": test_overlap}})

    return output
