import os

from data.conll03 import parse_raw_conll
from data.data_utils import build_vocab
from data.data_utils import iob2iobes
from data.loader import load_data
from global_vars import DATA_DIR, EMBEDDINGS_DIR
from utils.embeddings import trim_embeddings


def remap_tags(tags, mapping):
    return [[t.split("-")[0] + "-" + mapping["-".join(t.split("-")[1:])] if not t == "O" and not mapping["-".join(
        t.split("-")[1:])] == "O" else "O" for t in s] for s in tags]


if __name__ == "__main__":

    source_path = DATA_DIR + "wnut17/source/"
    original_path = DATA_DIR + "wnut17/original/"
    remapped_path = DATA_DIR + "wnut17/remapped/"

    if not os.path.exists(source_path):
        os.makedirs(source_path)
    if not os.path.exists(original_path):
        os.makedirs(original_path)
    if not os.path.exists(remapped_path):
        os.makedirs(remapped_path)

    ### Reformat Source data
    for split, file in zip(["train", "dev", "test"],
                           ["wnut17train.conll", "emerging.dev.conll", "emerging.test.annotated"]):
        words, tags, _ = parse_raw_conll(source_path + file, sep="\t")

        with open(original_path + "{}.words.txt".format(split), "w", encoding="utf8") as file:
            for sent in words:
                file.write("{}\n".format(" ".join(sent)))

        with open(original_path + "{}.iob.txt".format(split), "w", encoding="utf8") as file:
            for sent in tags:
                file.write("{}\n".format(" ".join(sent)))

        with open(original_path + "{}.iobes.txt".format(split), "w", encoding="utf8") as file:
            for sent in tags:
                file.write("{}\n".format(" ".join(iob2iobes(sent))))

    ### Compute vocabulary
    build_vocab(original_path)

    ### Trim glove embeddings
    word_vocab_path = original_path + "vocab.words.txt"
    embedding_path = EMBEDDINGS_DIR + "glove.840B/glove.840B.300d.txt"
    saving_path = original_path + "glove.840B.300d"
    trim_embeddings(word_vocab_path, embedding_path, saving_path, check_exists=False)

    ### Remap dataset
    data = load_data(original_path)
    tag2idx = data["vocab"]["tag"][0]

    mapping = {t: "O" for t in ["-".join(w.split("-")[1:]) for w in tag2idx.keys() if w[0] == "B"]}

    mapping["corporation"] = "ORG"
    mapping["location"] = "LOC"
    mapping["person"] = "PER"

    for split, file in zip(["train", "dev", "test"],
                           ["wnut17train.conll", "emerging.dev.conll", "emerging.test.annotated"]):
        words, tags, _ = parse_raw_conll(source_path + file, sep="\t")

        converted_tags = remap_tags(tags, mapping)
        with open(remapped_path + "{}.words.txt".format(split), "w", encoding="utf8") as file:
            for sent in words:
                file.write("{}\n".format(" ".join(sent)))

        with open(remapped_path + "{}.iob.txt".format(split), "w", encoding="utf8") as file:
            for sent in converted_tags:
                file.write("{}\n".format(" ".join(sent)))

        with open(remapped_path + "{}.iobes.txt".format(split), "w", encoding="utf8") as file:
            for sent in converted_tags:
                file.write("{}\n".format(" ".join(iob2iobes(sent))))

    ### Compute remapped vocabulary
    build_vocab(remapped_path)
