import json
import logging
import os
from copy import deepcopy

from data.data_iterator import init_iterators
from data.data_utils import extract_entities_corpus, compute_overlap
from data.loader import load_data
from embedders.embedder import Embedder, StackedEmbedder
from embedders.word_embedder import WordEmbedder
from global_vars import DATA_DIR
from models.model import EmbedderEncoderDecoder
from modules.decoders import SoftmaxDecoder, CRFDecoder
from modules.encoders import BiLSTMEncoder
from utils.logger_utils import set_logger
from utils.torch_utils import load_checkpoint
from utils.train_utils import check_and_format, parse_args


def intersect(vocab1, vocab2):
    return {i: w for i, w in vocab1.items() if w in set(vocab2.values())}


if __name__ == "__main__":
    parameters = parse_args()

    """ Check parameters and format """
    check_and_format(parameters, train=False)

    """ logger """
    print("Logging in {}".format(
        os.path.join(parameters["run_dir"], "test_ood_{}.log".format(parameters["ood_dataset"]))))
    set_logger(os.path.join(parameters["run_dir"], "test_ood_{}.log".format(parameters["ood_dataset"])))

    """ Load Data """
    logging.info("Loading data...")
    filenames = splits = ["train", "dev", "test"]

    # Load training dataset
    data_path = DATA_DIR + parameters["dataset"] + "/"
    train_data = load_data(data_path + "original/")
    train_iterators = init_iterators(train_data["data"], train_data["vocab"],
                                     batch_size=parameters["batch_size"],
                                     device=parameters["device"], shuffle_train=True)
    train_entities = extract_entities_corpus(train_data["data"]["train"])

    # Load ood test dataset
    ood_data_path = DATA_DIR + parameters["ood_dataset"] + "/"
    ood_data = load_data(ood_data_path + "remapped/", embedding_path=ood_data_path + "original/glove.840B.300d")

    test_vocab = deepcopy(train_data["vocab"])
    test_vocab["word"] = ood_data["vocab"]["word"]

    ood_iterators = init_iterators(ood_data["data"], test_vocab,
                                   batch_size=parameters["batch_size"],
                                   device=parameters["device"])

    """ Model """
    # Embedder

    embedders = [Embedder(mode,
                          w_embeddings=train_data["embeddings"],
                          params=parameters,
                          word_vocab=test_vocab["word"],
                          char_vocab=test_vocab["char"]) for mode in parameters["embedder"]]
    embedder = StackedEmbedder(embedders)

    # Encoder
    if parameters["encoder"] == "bilstm":
        encoder = BiLSTMEncoder(embedder.w_embed_dim, parameters["word_hidden_dim"],
                                bidirectional=True,
                                dropout=parameters["dropout"])
        output_dim = encoder.output_dim
    else:
        assert parameters["encoder"] == "map"
        encoder = None
        output_dim = embedder.w_embed_dim

    # Decoder
    if parameters["decoder"] == "crf":
        decoder = CRFDecoder(output_dim, train_data["vocab"]["tag"], dropout=parameters["dropout"])
    elif parameters["decoder"] == "softmax":
        decoder = SoftmaxDecoder(output_dim, train_data["vocab"]["tag"], dropout=parameters["dropout"])

    # Model
    model = EmbedderEncoderDecoder(embedder, encoder, decoder)
    model.to(parameters["device"])

    # Reload best checkpoint
    state = load_checkpoint(parameters["run_dir"] + "ner_best.pth.tar")
    model.load_state_dict(state["model"])

    # Change Glove Embedder
    if "word_embedder" in model.embedder.embedders.keys():
        logging.info("Changing embedder to OOD data and intersecting with trained embeddings...")

        ### Find words in OOD vocabulary in Train vocabulary
        inter_vocab = intersect(train_data["vocab"]["word"][1], ood_data["vocab"]["word"][1])

        ### Load trained embeddings values
        train_embeddings = model.embedder.embedders["word_embedder"].embedder.word_embeddings.weight

        ### Replace corresponding embeddings with trained values
        for train_idx, w in inter_vocab.items():
            ood_idx = ood_data["vocab"]["word"][0][w]
            ood_data["embeddings"][ood_idx] = train_embeddings[train_idx]

        ### Change embedder
        ood_embedder = WordEmbedder(test_vocab["word"], 300, w_embeddings=ood_data["embeddings"],
                                    freeze=True).to(parameters["device"])
        model.embedder.embedders.update({"word_embedder": ood_embedder})

    ### OOD Evaluation
    logging.info("Computing lexical overlap...")
    dev_overlap, _ = compute_overlap(ood_data["data"]["dev"]["words"], ood_data["data"]["dev"]["tags"], train_entities)
    test_overlap, _ = compute_overlap(ood_data["data"]["test"]["words"], ood_data["data"]["test"]["tags"],
                                      train_entities)

    logging.info("OOD Evaluation")
    logging.info("Dev eval")
    dev_preds, dev_loss, dev_scores = model.evaluate(ood_iterators["dev"], overlap=dev_overlap,
                                                     train_entities=train_entities)
    logging.info("Test eval")
    test_preds, test_loss, test_scores = model.evaluate(ood_iterators["test"], overlap=test_overlap,
                                                        train_entities=train_entities)

    with open(parameters["run_dir"] + "ner_ood_{}_dev_scores.json".format(parameters["ood_dataset"]), "w") as file:
        json.dump(dev_scores, file)
    with open(parameters["run_dir"] + "ner_ood_{}_test_scores.json".format(parameters["ood_dataset"]), "w") as file:
        json.dump(test_scores, file)
