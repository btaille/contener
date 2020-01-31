import os
import pickle

import numpy as np
from tqdm import tqdm


def add_oov_padding(w2idx, oov="<UNK>", pad="<PAD>"):
    """Add OOV and PAD tokens to w2idx dict."""
    w2idx = {w: idx + 2 for w, idx in w2idx.items()}
    w2idx.update({pad: 0, oov: 1})
    return w2idx


def add_start_stop(w2idx):
    w2idx.update({"<S>": len(w2idx)})
    w2idx.update({"</S>": len(w2idx)})
    return w2idx


def load_vocabulary(word_vocab_path, oov_padding=False, oov="<UNK>", pad="<PAD>", start_stop=False):
    """Load w2idx and idx2w dicts from vocab file path."""
    with open(word_vocab_path, "r", encoding="utf8") as file:
        w2idx = {line.strip(): idx for idx, line in enumerate(file)}

    if oov_padding:
        w2idx = add_oov_padding(w2idx, oov, pad)

    if start_stop:
        w2idx = add_start_stop(w2idx)

    idx2w = {v: k for (k, v) in w2idx.items()}

    return w2idx, idx2w


def load_embeddings(embedding_path, oov_padding=True):
    """Load embeddings np.array from saved path."""

    embeddings = np.load(embedding_path + "_embeddings.npz")["embeddings"]
    with open(embedding_path + "_w2idx.pickle", "rb") as file:
        w2idx = pickle.load(file)

    if oov_padding:
        oov = np.zeros((2, embeddings.shape[1]))
        embeddings = np.vstack([oov, embeddings])

    w2idx = add_oov_padding(w2idx, "<UNK>", "<PAD>")
    return w2idx, embeddings


def trim_embeddings(word_vocab_path, embedding_path, saving_path, embedding_dim=300, check_exists=True):
    """Trim pretrained embeddings to fixed vocabulary.

    Args:
        word_vocab_path (str) : path of word vocabulary file
        embedding_path (str) : path of pretrained word embeddings
        saving_path (str) : path to save trim embeddings in .npz format
        embedding_dim (int) : (default 300)
    """
    # Load word vocabulary
    if os.path.exists(saving_path + "_embeddings.npz") and check_exists:
        print("Saving files already exists")

    else:
        print("Loading vocabulary...")
        with open(word_vocab_path, "r", encoding="utf8") as file:
            w2idx = {line.strip(): idx for idx, line in enumerate(file)}
        size_vocab = len(w2idx)

        # Initialize random embeddings
        np.random.seed(0)
        embeddings = np.random.normal(scale=1e-3, size=(size_vocab, embedding_dim))

        print("Reading embedding file...")
        found = 0
        with open(embedding_path, "r", encoding="utf8") as file:
            for i, line in enumerate(tqdm(file)):
                line = line.strip().split()
                if not len(line) == embedding_dim + 1:
                    continue

                word, embedding = line[0], line[1:]
                if word in w2idx:
                    found += 1
                    embeddings[w2idx[word]] = embedding
        print("Word embeddings trimmed. Found {} vectors for {} words".format(found, size_vocab))

        np.savez_compressed(saving_path + "_embeddings.npz", embeddings=embeddings)
        with open(saving_path + "_w2idx.pickle", "wb") as file:
            pickle.dump(w2idx, file)


def load_glove(embedding_path, saving_path, embedding_dim=300):
    if os.path.exists(saving_path + "_embeddings.npz"):
        embeddings = np.load(saving_path + "_embeddings.npz")["embeddings"]
        with open(saving_path + "_w2idx.pickle", "rb") as file:
            w2idx = pickle.load(file)
    else:
        # Load word vocabulary
        idx2w = {}

        print("Counting lines")
        lines = 0
        with open(embedding_path, "r", encoding="utf8") as file:
            for line in tqdm(file):
                line = line.strip().split()
                if not len(line) == embedding_dim + 1:
                    continue
                lines += 1

        embeddings = np.zeros((lines, embedding_dim))
        print("Reading embedding file...")
        with open(embedding_path, "r", encoding="utf8") as file:
            for line in tqdm(file):
                line = line.strip().split()
                if not len(line) == embedding_dim + 1:
                    continue

                word, embedding = line[0], line[1:]
                idx = len(idx2w)
                idx2w[idx] = word
                embeddings[idx] = embedding

        w2idx = {v: k for k, v in idx2w.items()}

        with open(saving_path + "_w2idx.pickle", "wb") as file:
            pickle.dump(w2idx, file)
        np.savez_compressed(saving_path + "_embeddings.npz", embeddings=embeddings)

    return w2idx, embeddings
