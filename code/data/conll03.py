from allennlp import data
import pandas as pd
from tqdm import tqdm

def parse_raw_conll(path, sep=" ", word_column_idx=0, tag_column_idx=-1):
    docs = []
    sents, tags = [], []
    tmp_sent, tmp_tag = [], []
    with open(path, "r", encoding="utf8") as file:
        for i, line in enumerate(file):
            if not len(line.strip()) and len(tmp_sent):
                sents.append(tmp_sent)
                tags.append(tmp_tag)
                tmp_sent, tmp_tag = [], []

            elif line == "\n":
                pass

            elif line.split(sep)[0] == "-DOCSTART-":
                docs.append(len(sents))

            else:
                annotations = line.strip().split(sep)
                word = annotations[word_column_idx]
                tag = annotations[tag_column_idx]

                tmp_sent.append(word)
                tmp_tag.append(tag)

        if not len(line.strip()) and len(tmp_sent):
            sents.append(tmp_sent)
            tags.append(tmp_tag)

    return sents, tags, docs


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
