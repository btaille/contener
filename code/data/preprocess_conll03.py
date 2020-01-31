import os

from data.conll03 import parse_raw_conll
from data.data_utils import build_vocab
from data.data_utils import iob2iobes
from global_vars import DATA_DIR, EMBEDDINGS_DIR
from utils.embeddings import trim_embeddings
from utils.evaluation import correct_iob

if __name__ == "__main__":

    source_path = DATA_DIR + "conll03/source/"
    original_path = DATA_DIR + "conll03/original/"

    if not os.path.exists(source_path):
        os.makedirs(source_path)
    if not os.path.exists(original_path):
        os.makedirs(original_path)

    ### Reformat Source data
    for split, file in zip(["train", "dev", "test"],
                           ["eng.train", "eng.testa", "eng.testb"]):
        words, tags, _ = parse_raw_conll(source_path + file, sep="\t")

        ### Correct IOB tags in case eng.train ..etc.. have only I- tags
        tags = correct_iob(tags)

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
    trim_embeddings(word_vocab_path, embedding_path, saving_path)
