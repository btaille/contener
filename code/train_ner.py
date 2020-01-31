import json
import logging
import os

from torch import optim

from global_vars import DATA_DIR
from data.data_iterator import init_iterators
from data.data_utils import extract_entities_corpus, compute_overlap
from data.loader import load_data
from embedders.embedder import Embedder, StackedEmbedder
from models.model import EmbedderEncoderDecoder
from modules.decoders import CRFDecoder, SoftmaxDecoder
from modules.encoders import BiLSTMEncoder
from utils.logger_utils import set_logger
from utils.torch_utils import load_checkpoint
from utils.train_utils import check_and_format, parse_args, set_random_seed

if __name__ == "__main__":
    parameters = parse_args()

    """ Check parameters and format """
    check_and_format(parameters)

    """ Set random seeds """
    set_random_seed(parameters["seed"])

    """ logger """
    print("Logging in {}".format(os.path.join(parameters["run_dir"], "train.log")))
    set_logger(os.path.join(parameters["run_dir"], "train.log"))

    """ Load Data """
    logging.info("Loading data...")
    filenames = splits = ["train", "dev", "test"]

    # Load dataset
    data_path = DATA_DIR + parameters["dataset"] + "/"
    if parameters["dataset"] == "conll03":
        data = load_data(data_path + "original/")
    else:
        data = load_data(data_path + "remapped/", embedding_path=data_path + "original/glove.840B.300d")

    iterators = init_iterators(data["data"], data["vocab"],
                               batch_size=parameters["batch_size"],
                               device=parameters["device"], shuffle_train=True)

    # Compute lexical overlap
    logging.info("Computing lexical overlap...")
    train_entities = extract_entities_corpus(data["data"]["train"])
    dev_overlap, _ = compute_overlap(data["data"]["dev"]["words"], data["data"]["dev"]["tags"], train_entities)
    test_overlap, _ = compute_overlap(data["data"]["test"]["words"], data["data"]["test"]["tags"], train_entities)

    """ Model """
    logging.info("Initializing Model...")
    # Embedder
    embedders = [Embedder(mode,
                          w_embeddings=data["embeddings"],
                          params=parameters,
                          word_vocab=data["vocab"]["word"],
                          char_vocab=data["vocab"]["char"],
                          device=parameters["device"]) for mode in parameters["embedder"]]
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
        decoder = CRFDecoder(output_dim, data["vocab"]["tag"], dropout=parameters["dropout"])
    elif parameters["decoder"] == "softmax":
        decoder = SoftmaxDecoder(output_dim, data["vocab"]["tag"], dropout=parameters["dropout"])

    # Model
    model = EmbedderEncoderDecoder(embedder, encoder, decoder)
    model.to(parameters["device"])

    """ Optimization """
    # Set trainable parameters LR
    trainable_parameters = filter(lambda p: p.requires_grad, model.parameters())

    for module, name in zip([model, model.embedder, model.encoder, model.decoder],
                            ["model", "embedder", "encoder", "decoder"]):
        num_params = sum(param.numel() for param in module.parameters()) if module is not None else 0
        num_params_trainable = sum(
            param.numel() for param in module.parameters() if param.requires_grad) if module is not None else 0
        logging.info("{} / {} trainable parameters in {}".format(num_params_trainable, num_params, name))

    if parameters["embedder"] in ["bert-large", "bert-base"]:
        optimizer_params = [{"params": model.encoder.parameters()},
                            {"params": model.decoder.parameters()},
                            {"params": model.embedder.parameters(), "lr": 5e-5}]
    else:
        optimizer_params = [{"params": model.parameters()}]

    # Optimizer
    if parameters["optimizer"].lower() == "adam":
        optimizer = optim.Adam(optimizer_params, lr=parameters["lr"], betas=(0.9, 0.999), eps=1e-8,
                               weight_decay=0)
    elif parameters["optimizer"].lower() == "sgd":
        optimizer = optim.SGD(optimizer_params, lr=parameters["lr"])

    """ Train """
    logging.info("Training model in {}".format(parameters["run_dir"]))

    epoch_start = 0
    best_f1 = 0
    epochs_no_improv = 0
    best_score = None

    # Reload if exist
    if os.path.exists(parameters["run_dir"] + "ner_checkpoint.pth.tar"):
        logging.info("Reload model...")
        ner_checkpoint = load_checkpoint(parameters["run_dir"] + "ner_checkpoint.pth.tar")
        epoch_start = ner_checkpoint["epoch"]
        epochs_no_improv = ner_checkpoint["epochs_no_improv"]
        best_score = ner_checkpoint["scores"]
        best_f1 = ner_checkpoint["scores"]["ALL"]["f1"]

        optimizer.load_state_dict(ner_checkpoint["optimizer"])

        model.load_state_dict(ner_checkpoint["model"])

    # Train Loop
    model.train_loop(iterators, optimizer, parameters["run_dir"],
                     overlap=dev_overlap,
                     train_entities=train_entities,
                     epochs=parameters["epochs"], patience=parameters["patience"], epoch_start=epoch_start,
                     epochs_no_improv=epochs_no_improv, min_epochs=parameters["min_epochs"],
                     best_f1=best_f1, train_key="train")

    # Test
    best_checkpoint = load_checkpoint(parameters["run_dir"] + "ner_best.pth.tar")
    model.load_state_dict(best_checkpoint["model"])

    logging.info("Dev eval")
    dev_preds, dev_loss, dev_scores = model.evaluate(iterators["dev"],
                                                     overlap=dev_overlap,
                                                     train_entities=train_entities)
    logging.info("Test eval")
    test_preds, test_loss, test_scores = model.evaluate(iterators["test"],
                                                        overlap=test_overlap,
                                                        train_entities=train_entities)

    # Write scores
    with open(parameters["run_dir"] + "ner_dev_scores.json", "w") as file:
        json.dump(dev_scores, file)
    with open(parameters["run_dir"] + "ner_test_scores.json", "w") as file:
        json.dump(test_scores, file)
