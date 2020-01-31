import logging
import os
import re
from collections import Counter

import numpy as np

from data.data_iterator import ids2seq


def load_file(path):
    """Load file formated with one sentence per line and tokens separated by spaces."""
    with open(path, "r", encoding="utf8") as file:
        outputs = [line.strip().split() for line in file]
    return outputs


def iob2iobes(iob):
    """Converts a list of IOB tags to IOBES scheme."""
    iobes = []
    tags = iob + ["O"]

    for i in range(len(tags) - 1):
        tag, next_tag = tags[i: i + 2]

        if tag == "O":
            iobes.append("O")
        else:
            if tag[0] == "B":
                if next_tag[0] in "OB" or not "-".join(next_tag.split("-")[1:]) == "-".join(tag.split("-")[1:]):
                    iobes.append("S-" + "-".join(tag.split("-")[1:]))
                else:
                    iobes.append(tag)
            elif tag[0] == "I":
                if next_tag[0] == "O" or not "-".join(next_tag.split("-")[1:]) == "-".join(tag.split("-")[1:]):
                    iobes.append("E-" + "-".join(tag.split("-")[1:]))
                else:
                    iobes.append(tag)

    return iobes


def iobes2iob(iobes):
    """Converts a list of IOBES tags to IOB scheme."""
    dico = {pfx: pfx for pfx in "IOB"}
    dico.update({"S": "B", "E": "I"})
    return [dico[t[0]] + t[1:] if not t == "O" else "O" for t in iobes]


def iob_scheme(tags, iobes=True):
    """ Transform tag sequence without any scheme into IOB or IOBES scheme """
    iob = []
    tags = ["O"] + tags + ["O"]

    for i in range(1, len(tags) - 1):
        prev_tag, tag, next_tag = tags[i - 1: i + 2]

        if not tag == "O":
            if tag not in [prev_tag, next_tag]:
                if iobes:
                    iob.append("S-" + tag)
                else:
                    iob.append("B-" + tag)
            elif not tag == prev_tag:
                iob.append("B-" + tag)
            elif not tag == next_tag:
                if iobes:
                    iob.append("E-" + tag)
                else:
                    iob.append("I-" + tag)
            else:
                iob.append("I-" + tag)
        else:
            iob.append("O")

    return iob


def build_word_vocab(data_path, filenames, mincount=1, whole_dataset=False):
    """Computes word vocabulary of several files and save it to "vocab.words.txt".

    Args:
        data_path (str) : Path of the folder containing the sentence files named "[filename].words.txt"
        filenames (list) : list of filenames (ex : ["train"] if file is named "train.words.txt")
        mincount (int) : minimal number of occurences to consider (default:1)

    Returns:
        set of words kept
    """
    word_counter = Counter()
    for fn in filenames:
        vocab = Counter()
        with open(data_path + fn + ".words.txt", "r", encoding="utf8") as file:
            for line in file:
                word_counter.update(line.strip().split())
                vocab.update(line.strip().split())

            vocab = {w for w, c in vocab.items() if c >= mincount}
            with open(data_path + "vocab.{}.words.txt".format(fn), "w", encoding="utf8") as file:
                for w in sorted(list(vocab)):
                    file.write(w + "\n")

    word_vocab = {w for w, c in word_counter.items() if c >= mincount}

    if whole_dataset:
        with open(data_path + "vocab.words.txt", "w", encoding="utf8") as file:
            for w in sorted(list(word_vocab)):
                file.write(w + "\n")

    print('Word vocabulary built. Kept {} out of {}'.format(len(word_vocab), len(word_counter)))

    return word_vocab


def build_char_vocab(word_vocab, data_path, save_file="vocab.chars.txt"):
    """Computes char vocabulary given a word vocabulary path and save it to "vocab.chars.txt".

        Args:
            word_vocab (set) : set of words in the vocabulary
            data_path (str) : Path of the folder containing "vocab.words.txt"

        Returns:
            set of chars
    """
    char_vocab = set()
    for w in word_vocab:
        char_vocab.update(w)

    with open(os.path.join(data_path, save_file), "w", encoding="utf8") as file:
        for w in sorted(list(char_vocab)):
            file.write(w + "\n")

    print('Char vocabulary built. Found {} chars'.format(len(char_vocab)))

    return char_vocab


def build_tag_vocab(data_path, filenames, iobes=True):
    """Computes tag vocabulary of several files and save it to "vocab.[scheme].txt".

    Args:
        data_path (str) : Path of the folder containing the tag files named "[filename].[scheme].txt"
        filenames (list) : list of filenames (ex : ["train"] if file is named "train.iobes.txt")
        iobes (bool) : whether to use IOBES scheme (default:True) else IOB scheme
    """

    if iobes:
        ext = ".iobes.txt"
    else:
        ext = ".iob.txt"

    tag_vocab = set()
    for fn in filenames:
        with open(data_path + fn + ext, "r", encoding="utf8") as file:
            for line in file:
                tag_vocab.update(line.strip().split())

    with open(data_path + "vocab" + ext, "w", encoding="utf8") as file:
        for t in sorted(list(tag_vocab)):
            file.write(t + "\n")

    print('Tag vocabulary built. Found {} tags'.format(len(tag_vocab)))


def build_vocab(data_path, filenames=["train", "dev", "test"], mincount=1):
    """Build word, char and tag vocabulary in both IOB and IOBES."""
    word_vocab = build_word_vocab(data_path, filenames, mincount, whole_dataset=True)
    build_char_vocab(word_vocab, data_path)
    train_word_vocab = build_word_vocab(data_path, ["train"], mincount)
    build_char_vocab(train_word_vocab, data_path, save_file="vocab.train.chars.txt")
    build_tag_vocab(data_path, filenames, iobes=True)
    build_tag_vocab(data_path, filenames, iobes=False)


def extract_entities(words, iob_tags):
    """ Retrieve entities, types and spans given a tokenized sentence and IOB tags

    Args:
        words (list) : list of word strings
        iob_tags (list) : list of IOB tags

    Returns:
        entities (list) : list of words in the extracted entities
        types (list) : list of types of the extracted entities
        indices (list) : list of spans (begin idx, end idx) of the entities
    """
    entities = []
    types = []
    indices = []
    tmp_ent = None
    tmp_indices = None
    tmp_type = "O"
    for i, (w, t) in enumerate(zip(words, iob_tags)):
        if t[0] == "B":
            if tmp_ent is not None:
                entities.append(tmp_ent)
                indices.append(tmp_indices)
                types.append(tmp_type)
            tmp_ent = w
            tmp_type = "-".join(t.split("-")[1:])
            tmp_indices = [i]

        elif t[0] == "O":
            if tmp_ent is not None:
                entities.append(tmp_ent)
                indices.append(tmp_indices)
                types.append(tmp_type)
            tmp_ent = None
            tmp_type = None
            tmp_indices = None

        elif t[0] == "I":
            if "-".join(t.split("-")[1:]) == tmp_type and i == tmp_indices[-1] + 1:
                tmp_ent += " " + w
                tmp_indices += [i]
            else:
                if tmp_ent is not None:
                    entities.append(tmp_ent)
                    indices.append(tmp_indices)
                    types.append(tmp_type)
                tmp_ent = w
                tmp_type = "-".join(t.split("-")[1:])
                tmp_indices = [i]

    if tmp_ent is not None:
        entities.append(tmp_ent)
        indices.append(tmp_indices)
        types.append(tmp_type)

    return list(zip(entities, types, indices))


def extract_iob(iob_tags):
    """ Retrieve types and spans given a tokenized sentence and IOB tags

    Args:
        iob_tags (list) : list of IOB tags

    Returns:
        types (list) : list of types of the extracted entities
        indices (list) : list of spans (begin idx, end idx) of the entities
    """
    types = []
    indices = []

    tmp_indices = None
    tmp_type = "O"
    for i, t in enumerate(iob_tags):
        if t[0] == "B":
            if tmp_indices is not None:
                indices.append(tmp_indices)
                types.append(tmp_type)
            tmp_type = "-".join(t.split("-")[1:])
            tmp_indices = [i]

        elif t[0] == "O":
            if tmp_indices is not None:
                indices.append(tmp_indices)
                types.append(tmp_type)
            tmp_type = None
            tmp_indices = None

        elif t[0] == "I":
            if "-".join(t.split("-")[1:]) == tmp_type and i == tmp_indices[-1] + 1:
                tmp_indices += [i]
            else:
                if tmp_indices is not None:
                    indices.append(tmp_indices)
                    types.append(tmp_type)
                tmp_type = "-".join(t.split("-")[1:])
                tmp_indices = [i]

    if tmp_indices is not None:
        indices.append(tmp_indices)
        types.append(tmp_type)

    return list(zip(types, indices))


def extract_entities_corpus(data, unique=True):
    entities = {}
    if "words" in data.keys():
        for w, t in zip(data["words"], data["tags"]):
            ents = extract_entities(w.split(), iobes2iob(t.split()))

            for e, t, s in ents:
                if not t in entities.keys():
                    entities[t] = [e]
                else:
                    entities[t].append(e)
        if unique:
            for type in entities.keys():
                entities[type] = set(entities[type])
    else:
        for split in data.keys():
            if not len(entities):
                entities = extract_entities_corpus(data[split])
            else:
                to_append = extract_entities_corpus(data[split])
                assert set(to_append.keys()) == set(entities.keys())
                for k in entities.keys():
                    entities[k].extend(to_append[k])

    return entities


def partial_match(ent, entities, stop_words=None):
    word_set = set(np.concatenate([ent.split() for ent in set(entities)]))

    if stop_words is None:
        stop_words = set(". , : ; ? ! ( ) \" ' % - the on 's of a an".split())
    else:
        stop_words = set(". , : ; ? ! ( ) \" ' % - the on 's of a an".split() + stop_words)

    for e in ent.split():
        if e in word_set and not e.lower() in stop_words:
            return True
    return False


def compute_overlap(words, tags, train_entities, stop_words=None, verbose=False):
    overlaps = []
    splits = ["EXACT", "PARTIAL", "NEW"]
    overlap_count = {ent_type: {split: 0 for split in splits} for ent_type in train_entities.keys()}

    for i, (w, t) in enumerate(zip(words, tags)):
        t = iobes2iob(t.split())
        w = w.split()

        entities = extract_entities(w, t)

        current_overlap = ["O"] * len(w)

        for ent, typ, span in entities:
            if ent in train_entities[typ]:
                for idx in span:
                    current_overlap[idx] = "EXACT"
                overlap_count[typ]["EXACT"] += 1

            elif partial_match(ent, train_entities[typ], stop_words=stop_words):
                for idx in span:
                    current_overlap[idx] = "PARTIAL"
                overlap_count[typ]["PARTIAL"] += 1
            else:
                for idx in span:
                    current_overlap[idx] = "NEW"
                overlap_count[typ]["NEW"] += 1

        overlaps.append(current_overlap)

    for ent_type in train_entities.keys():
        n_mentions = sum([overlap_count[ent_type][split] for split in splits])

        if verbose:
            logging.info(ent_type)
            logging.info("{} mentions".format(n_mentions))
            logging.info("{} exact overlapping".format(overlap_count[ent_type]["EXACT"]))
            logging.info("{} %".format(100 * overlap_count[ent_type]["EXACT"] / n_mentions))
            logging.info("{} partial overlapping".format(overlap_count[ent_type]["PARTIAL"]))
            logging.info("{} %".format(100 * overlap_count[ent_type]["PARTIAL"] / n_mentions))
            logging.info("{} unseen".format(overlap_count[ent_type]["NEW"]))
            logging.info("{} %".format(100 * overlap_count[ent_type]["NEW"] / n_mentions))
            logging.info("")

    return overlaps, overlap_count