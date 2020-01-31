import argparse
import json
import os
import random
from collections import OrderedDict

import numpy as np
import torch

from global_vars import RUN_DIR


def set_random_seed(seed, cuda=True):
    np.random.seed(seed)  # cpu vars
    torch.manual_seed(seed)  # cpu  vars
    random.seed(seed)  # Python

    if cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # gpu vars
        torch.backends.cudnn.deterministic = True  # needed
        torch.backends.cudnn.benchmark = False


def check_and_format(parameters, train=True, prefix=None):
    """Assert parameters are a valid set and format training folder."""
    # Check parameters
    assert parameters["optimizer"].lower() in ["sgd", "adam"]
    if "char" in parameters["embedder"]:
        assert parameters["char_pool"] in ["last", "avg", "max"]
    assert parameters["decoder"] in ["softmax", "crf"]
    assert parameters["scheme"] in ["iob", "iobes"]

    if "cuda" in parameters["device"]:
        assert torch.cuda.is_available(), "CUDA not available"

    # Format
    if not "run_dir" in parameters.keys():
        param_str = "{}_{}_{}".format(parameters["dataset"], "-".join(sorted(parameters["embedder"])),
                                      parameters["encoder"])

        param_str += "_{}_{}_lr-{}_bs-{}_d-{}".format(
            *[parameters[k] for k in "decoder optimizer lr batch_size dropout".split()])

        if "glove" in parameters["embedder"]:
            param_str += "_glove-{}_freeze-{}".format(
                *[parameters[k] for k in "glove_word_embedding_dim freeze_embeddings".split()])
        if "random" in parameters["embedder"]:
            param_str += "_rand-{}".format(parameters["word_embedding_dim"])

        if "char" in parameters["embedder"]:
            param_str += "_char-{}_ch-{}_cp-{}".format(
                *[parameters[k] for k in "char_embedding_dim char_hidden_dim char_pool".split()])

        # Encoder Specific
        if parameters["encoder"] == "bilstm":
            param_str += "_hidden-{}".format(parameters["word_hidden_dim"])
            if parameters["bilstm_layers"] > 1:
                param_str += "-l-{}".format(parameters["bilstm_layers"])

        if parameters["device"] == "cpu":
            param_str += "_cpu"

        if parameters["scheme"] == "iob":
            param_str += "_iob"

        if "gradient_clipping" in parameters.keys() and parameters["gradient_clipping"]:
            param_str += "_gc-{}".format(parameters["gradient_clipping"])

        if "suffix" in parameters.keys() and not parameters["suffix"] == "":
            param_str += "_{}".format(parameters["suffix"])

        parameters["run_dir"] = "{}{}/seed_{}/".format(RUN_DIR, param_str, parameters["seed"])

    if prefix is not None:
        path = parameters["run_dir"].split("/")
        parameters["run_dir"] = "/".join(path[:-3]) + "/{}_".format(prefix) + "/".join(path[-3:])

    if not os.path.exists(parameters["run_dir"]):
        os.makedirs(parameters["run_dir"])
        with open(parameters["run_dir"] + "parameters.json", "w") as file:
            json.dump(parameters, file)

    elif train and os.path.exists(parameters["run_dir"] + "ner_test_scores.json"):
        assert False, "Run already launched"

    elif not train and os.path.exists(parameters["run_dir"] + "ner_ood_test_scores.json"):
        assert False, "Test OOD already launched"


def parse_args(arg_list=None):
    """ Argument Parser """
    parser = argparse.ArgumentParser()

    # Config in json
    parser.add_argument("-cf", "--config_file", help="config json file", default="none")

    # From command_line
    parser.add_argument("-ds", "--dataset", help="dataset", default="conll03")
    parser.add_argument("-emb", "--embedder", help="embedder list", nargs="+", default=["glove"])
    parser.add_argument("-enc", "--encoder", help="encoder list", default="bilstm")
    parser.add_argument("-dec", "--decoder", help="decoder", default="crf")

    # For test_ood.py
    parser.add_argument("-ood_ds", "--ood_dataset", help="out of domain dataset", default=None)

    # train hyperparameters
    parser.add_argument("-s", "--seed", type=int, help="torch manual random seed", default=0)
    parser.add_argument("-ep", "--epochs", type=int, help="number of epochs", default=100)
    parser.add_argument("-p", "--patience", type=int, help="patience", default=5)
    parser.add_argument("-min", "--min_epochs", type=int, help="minimum epochs", default=10)

    parser.add_argument("-lr", "--learning_rate", type=float, help="learning rate", default=1e-3)
    parser.add_argument("-opt", "--optimizer", help="optimizer", default="adam")
    parser.add_argument("-bs", "--batch_size", type=int, help="batch size", default=64)
    parser.add_argument("-d", "--dropout", type=float, help="dropout", default=0.5)
    parser.add_argument("-dev", "--device", help="pytorch device", default="cuda")
    parser.add_argument("-gc", "--gradient_clipping", type=float, help="gradient clipping", default=0)

    # char embedder
    parser.add_argument("-ce", "--char_dim", type=int, help="dimension of char embeddings", default=100)
    parser.add_argument("-ch", "--char_hidden", type=int, help="dimension of char hidden layer", default=25)
    parser.add_argument("-cp", "--char_pool", help="pooling for the char-level encoder", default="last")

    # word embedder
    parser.add_argument("-gwe", "--glove_word_dim", type=int, help="dimension of glove embeddings", default=300)
    parser.add_argument("-we", "--word_dim", type=int, help="dimension of random word embeddings", default=300)
    parser.add_argument("-f", "--freeze", type=int, help="freeze glove embeddings", default=0)

    # BiLSTM encoder
    parser.add_argument("-wh", "--word_hidden", type=int, help="dimension of word hidden layer", default=100)
    parser.add_argument("-ll", "--bilstm_layers", type=int, help="num bilstm layers", default=1)

    # NER decoder
    parser.add_argument("-ts", "--tag_scheme", help="tagging scheme", default="iobes")

    # Suffix
    parser.add_argument("-sfx", "--suffix", help="suffix to run dir", default="")

    """ Parameters """
    if arg_list is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(arg_list)

    if not args.config_file == "none":
        return load_parameters(args.config_file)

    parameters = OrderedDict()

    parameters["dataset"] = args.dataset

    parameters["embedder"] = args.embedder
    parameters["encoder"] = args.encoder
    parameters["decoder"] = args.decoder
    parameters["scheme"] = args.tag_scheme

    # ood dataset
    if args.ood_dataset is not None:
        parameters["ood_dataset"] = args.ood_dataset

    # train hyperparameters
    parameters["seed"] = args.seed
    parameters["epochs"] = args.epochs
    parameters["patience"] = args.patience
    parameters["min_epochs"] = args.min_epochs

    parameters["lr"] = args.learning_rate
    parameters["optimizer"] = args.optimizer.lower()
    parameters["batch_size"] = args.batch_size
    parameters["dropout"] = args.dropout

    parameters["device"] = args.device
    parameters["gradient_clipping"] = args.gradient_clipping

    # char embedder
    if "char" in parameters["embedder"]:
        parameters["char_embedding_dim"] = args.char_dim
        parameters["char_hidden_dim"] = args.char_hidden
        parameters["char_pool"] = args.char_pool

    # word embedder
    if "glove" in parameters["embedder"]:
        parameters["glove_word_embedding_dim"] = args.glove_word_dim
        parameters["freeze_embeddings"] = args.freeze

    if "random" in parameters["embedder"]:
        parameters["word_embedding_dim"] = args.word_dim

    ### Encoders ###
    # BiLSTM
    if "bilstm" in parameters["encoder"]:
        parameters["word_hidden_dim"] = args.word_hidden
        parameters["bilstm_layers"] = args.bilstm_layers

    parameters["suffix"] = args.suffix

    return parameters


def load_parameters(path):
    assert os.path.exists(path)
    with open(path, "r") as file:
        parameters = json.load(file)
    parameters["run_dir"] = os.path.abspath(os.path.join(path, os.pardir))
    return parameters
