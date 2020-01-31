import argparse
import os
from allennlp import data
import pandas as pd
from tqdm import tqdm

from global_vars import DATA_DIR, EMBEDDINGS_DIR
from data.data_utils import build_vocab, iob2iobes, load_file
from data.loader import load_data
from utils.embeddings import trim_embeddings


def remap_tags(tags, mapping):
    return [[t.split("-")[0] + "-" + mapping["-".join(t.split("-")[1:])] if not t == "O" and not mapping["-".join(
        t.split("-")[1:])] == "O" else "O" for t in s] for s in tags]


def load_ontonotes(path, chunk_size=100):
    ontonotes_reader = data.dataset_readers.dataset_utils.Ontonotes()
    iterator = ontonotes_reader.dataset_iterator(path)

    dfs = []
    df = pd.DataFrame()

    for i, sent in enumerate(tqdm(iterator)):
        docid = sent.document_id
        sid = sent.sentence_id
        words = sent.words
        pos = sent.pos_tags
        ner = sent.named_entities
        coref = sent.coref_spans

        if i % chunk_size == 0:
            if len(df):
                dfs.append(df)
                df = pd.DataFrame()

        df = df.append({"docid": docid,
                        "sid": sid,
                        "words": words,
                        "pos": pos,
                        "ner": ner,
                        "coref": coref},
                       ignore_index=True)

    if len(df):
        dfs.append(df)

    return pd.concat(dfs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-p", "--path", help="conll-12 folder path", default=DATA_DIR + "conll-2012/")

    args = parser.parse_args()
    conll_path = args.path

    source_path = DATA_DIR + "ontonotes/source/"
    original_path = DATA_DIR + "ontonotes/original/"
    remapped_path = DATA_DIR + "ontonotes/remapped/"

    if not os.path.exists(source_path):
        os.makedirs(source_path)
    if not os.path.exists(original_path):
        os.makedirs(original_path)
    if not os.path.exists(remapped_path):
        os.makedirs(remapped_path)

    ### Load Ontonotes from conll-2012 folder
    train_df = load_ontonotes(os.path.join(conll_path, "v4/data/train/data/english/"))
    dev_df = load_ontonotes(os.path.join(conll_path, "v4/data/development/data/english/"))
    test_df = load_ontonotes(os.path.join(conll_path, "v4/data/test/data/english/"))

    ### Remove "pt" genre
    train_df = train_df[~train_df.docid.str.contains("pt/")]
    dev_df = dev_df[~dev_df.docid.str.contains("pt/")]
    test_df = test_df[~test_df.docid.str.contains("pt/")]

    ### Print statistics
    for df, split in zip([train_df, dev_df, test_df], "train dev test".split()):
        print()
        print(split)
        print("documents :", len(set(list(df.docid))))
        print("sents :", len(list(df.docid)))
        print("entities :", sum([len([ne for ne in s if ne[0] == "B"]) for s in list(df.ner)]))
        print("tokens :", sum([len(s) for s in list(df.words)]))

    ### Save DataFrames in source
    train_df.to_csv(source_path + "train.conll12.csv")
    dev_df.to_csv(source_path + "dev.conll12.csv")
    test_df.to_csv(source_path + "test.conll12.csv")

    ### Reformat files
    for df, split in zip([train_df, dev_df, test_df], "train dev test".split()):
        with open(original_path + "{}.words.txt".format(split), "w", encoding="utf8") as file:
            for i, row in df.iterrows():
                file.write("{}\n".format(" ".join(row["words"])))

        with open(original_path + "{}.iob.txt".format(split), "w", encoding="utf8") as file:
            for i, row in df.iterrows():
                file.write("{}\n".format(" ".join(row["ner"])))

        iob = load_file(original_path + "{}.iob.txt".format(split))

        with open(original_path + "{}.iobes.txt".format(split), "w", encoding="utf8") as file:
            for tag in iob:
                file.write("{}\n".format(" ".join(iob2iobes(tag))))

    ### Compute vocabulary
    build_vocab(original_path)

    ### Trim glove embeddings
    word_vocab_path = original_path + "vocab.words.txt"
    embedding_path = EMBEDDINGS_DIR + "glove.840B/glove.840B.300d.txt"
    saving_path = original_path + "glove.840B.300d"
    trim_embeddings(word_vocab_path, embedding_path, saving_path, check_exists=False)

    ### Remap dataset
    data = load_data(original_path, scheme="iob")
    tag2idx = data["vocab"]["tag"][0]

    mapping = {t: "O" for t in ["-".join(w.split("-")[1:]) for w in tag2idx.keys() if w[0] == "B"]}

    mapping["PERSON"] = "PER"
    mapping["ORG"] = "ORG"
    mapping["LOC"] = "LOC"
    mapping["GPE"] = "LOC"
    mapping["LANGUAGE"] = "MISC"
    mapping["NORP"] = "MISC"

    for split in ["train", "dev", "test"]:
        words = [s.split() for s in data["data"][split]["words"]]
        tags = [s.split() for s in data["data"][split]["tags"]]

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
