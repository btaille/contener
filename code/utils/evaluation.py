import copy
import logging
from collections import OrderedDict

import numpy as np

from data.data_utils import extract_iob, iobes2iob, extract_entities, partial_match


def sent_eval(pred_sent, tag_sent, entities):
    """Computes TP, FP and FN scores of IOB tags.
        
    Args:
        pred_sent (list) : list of IOB prediction strings
        tag_sent (list) : list of IOB ground truth strings
        entities (list) : list of possible entity types
        
    Returns:
        dict : number of TP, FP and FN per entity type
    """
    scores = {ent: {"tp": 0, "fp": 0, "fn": 0} for ent in entities}
    pred_ents = extract_iob(pred_sent)
    tag_ents = extract_iob(tag_sent)

    for ent in entities:
        scores[ent]["tp"] = len([e for e in pred_ents if e[0] == ent and e in tag_ents])
        scores[ent]["fn"] = len([e for e in tag_ents if e[0] == ent and e not in pred_ents])
        scores[ent]["fp"] = len([e for e in pred_ents if e[0] == ent and e not in tag_ents])
        assert scores[ent]["tp"] + scores[ent]["fp"] == len([e for e in pred_ents if e[0] == ent])
    return scores


def compute_accuracy(preds, tags):
    """Computes accuracy in any tagging scheme.
        
    Args:
        preds (list) : list of list of prediction strings
        tags (list) : list of list of ground truth strings
        
    Returns:
        float : accuracy (in %)
    """
    n_tokens = sum([len(tag) for tag in tags])
    correct_preds = 0
    for pred_sent, tag_sent in zip(preds, tags):
        correct_preds += sum([1 for p, t in zip(pred_sent, tag_sent) if p == t])

    return 100 * correct_preds / n_tokens


def correct_iob_sent(iob_sent, correct_b=False):
    """Correct IOB tagging scheme.

    Args:
        tags (list) : list of list of tag strings
        correct_b (bool) : whether to correct B-tag following B-tag or I-tag (default=False)
    Returns:
        list : list of list of corrected tags
    """
    corrected = copy.copy(iob_sent)
    prev_t = "O"
    for i, t in enumerate(iob_sent):

        if t[0] == "I":
            if prev_t == "O" or not "-".join(prev_t.split("-")[1:]) == "-".join(t.split("-")[1:]):
                corrected[i] = "B-" + t.split("-")[1]

        elif correct_b and t[0] == "B" and not prev_t == "O" and "-".join(prev_t.split("-")[1:]) == "-".join(
                t.split("-")[1:]):
            corrected[i] = "I-" + "-".join(t.split("-")[1:])

        prev_t = t

    return corrected


def correct_iob(tags, correct_b=False):
    """Correct IOB tagging scheme.
    
    Args:
        tags (list) : list of list of tag strings
        correct_b (bool) : whether to correct B-tag following B-tag or I-tag (default=False)
    Returns:
        list : list of list of corrected tags
    """
    return [correct_iob_sent(iob_sent, correct_b) for iob_sent in tags]


def compute_score(preds, tags, entities=["LOC", "MISC", "ORG", "PER"], scheme="iobes", correct=False):
    """Evaluate tag predictions.
    
    Args:
        preds (list) : list of list of prediction strings
        tags (list) : list of list of ground truth strings
        entities (list) : list of possible entity types (default : CoNLL03 entity types)
        scheme (str) : preds and tags tagging scheme (default : "iob")
        
    Returns:
        dict : per entity and global TP, FP, FN, precision, recall, f1
    """
    if scheme == "iobes":
        preds = [iobes2iob(s) for s in preds]
        tags = [iobes2iob(s) for s in tags]
    if correct:
        preds = correct_iob(preds)

    scores = {ent: {"tp": 0, "fp": 0, "fn": 0} for ent in entities + ["ALL"]}

    # Count True Positives / False Positives / False Negative
    n_tokens = sum([len(tag) for tag in tags])
    n_phrases = sum([len([t for t in tag if t[0] in ["B"]]) for tag in tags])
    n_found = sum([len([t for t in tag if t[0] in ["B"]]) for tag in preds])

    for pred_sent, tag_sent in zip(preds, tags):
        add_scores = sent_eval(pred_sent, tag_sent, entities)
        for ent in entities:
            for k in scores[ent].keys():
                scores[ent][k] += add_scores[ent][k]

    # Compute per entity Precision / Recall / F1
    for ent in scores.keys():
        if scores[ent]["tp"]:
            scores[ent]["p"] = 100 * scores[ent]["tp"] / (scores[ent]["fp"] + scores[ent]["tp"])
            scores[ent]["r"] = 100 * scores[ent]["tp"] / (scores[ent]["fn"] + scores[ent]["tp"])
        else:
            scores[ent]["p"], scores[ent]["r"] = 0, 0

        if not scores[ent]["p"] + scores[ent]["r"] == 0:
            scores[ent]["f1"] = 2 * scores[ent]["p"] * scores[ent]["r"] / (scores[ent]["p"] + scores[ent]["r"])
        else:
            scores[ent]["f1"] = 0

    # print summary
    tp = sum([scores[ent]["tp"] for ent in entities])
    fp = sum([scores[ent]["fp"] for ent in entities])
    fn = sum([scores[ent]["fn"] for ent in entities])

    accuracy = compute_accuracy(preds, tags)
    if tp:
        precision = 100 * tp / (tp + fp)
        recall = 100 * tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)

    else:
        precision, recall, f1 = 0, 0, 0

    scores["ALL"]["p"] = precision
    scores["ALL"]["r"] = recall
    scores["ALL"]["f1"] = f1
    scores["ALL"]["tp"] = tp
    scores["ALL"]["fp"] = fp
    scores["ALL"]["fn"] = fn

    logging.info(
        "processed {} tokens with {} phrases; found: {} phrases; correct: {}.".format(n_tokens, n_phrases, n_found, tp))
    logging.info(
        "accuracy: {:.2f}; \tTP: {};\tFP: {};\tFN: {};\tprecision: {:.2f};\trecall: {:.2f};\tf1: {:.2f}.".format(
            accuracy,
            scores["ALL"]["tp"],
            scores["ALL"]["fp"],
            scores["ALL"]["fn"],
            precision,
            recall,
            f1))
    for ent in entities:
        logging.info("\t\t{}: \tTP: {};\tFP: {};\tFN: {};\tprecision: {:.2f};\trecall: {:.2f};\tf1: {:.2f};\t{}".format(
            ent,
            scores[ent]["tp"],
            scores[ent]["fp"],
            scores[ent]["fn"],
            scores[ent]["p"],
            scores[ent]["r"],
            scores[ent]["f1"],
            scores[ent]["tp"] +
            scores[ent][
                "fp"]))

    return scores


def add_score(writer, score, n_iter, task="ner"):
    """Add score results to Tensorboard writer"""
    writer.add_scalars("{}/f1".format(task), {ent: score[ent]["f1"] for ent in score.keys()}, n_iter)
    writer.add_scalars("{}/p".format(task), {ent: score[ent]["p"] for ent in score.keys()}, n_iter)
    writer.add_scalars("{}/r".format(task), {ent: score[ent]["r"] for ent in score.keys()}, n_iter)


### Split CoNLL Eval script with Train Overlap Split
def partial_match(ent, entities_set, stop_words=None):
    word_set = set(np.concatenate([ent.split() for ent in entities_set]))

    if stop_words is None:
        stop_words = set(". , : ; ? ! ( ) \" ' % - the on 's of a an".split())
    else:
        stop_words = set(". , : ; ? ! ( ) \" ' % - the on 's of a an".split() + stop_words)

    for e in ent.split():
        if e in word_set and not e.lower() in stop_words:
            return True
    return False


def positive_overlap(pred_ents, train_entities):
    """Splits pred_ents into EXACT, PARTIAL and UNSEEN

    Args:
        pred_ents (list) : [(entity, type, span)]
    Returns:
        list : split of each mention in pred_ent
    """
    overlap = []

    for e, t, _ in pred_ents:
        if e in train_entities[t]:
            overlap.append("EXACT")
        elif partial_match(e, train_entities[t]):
            overlap.append("PARTIAL")
        else:
            overlap.append("NEW")

    return overlap


def sent_eval_overlap(sent, pred_sent, tag_sent, overlap_sent, train_entities):
    """Computes TP, FP and FN scores of IOB tags with distinction on seen or unseen entities.
        
    Args:
        sent (list) : list of words.
        pred_sent (list) : list of IOB prediction strings
        tag_sent (list) : list of IOB ground truth strings
        overlap_sent (list) : list of overlap strings ("NEW", "PARTIAL", "EXACT")
        entities (list) : list of possible entity types
        
    Returns:
        dict : per entity and per seen/partially/unseen split number of TP, FP and FN 
    """
    splits = ["NEW", "PARTIAL", "EXACT"]
    entities = list(train_entities.keys())

    scores = {ent: {split: {"tp": 0, "fp": 0, "fn": 0} for split in splits} for ent in entities}

    pred_ents = extract_entities(sent, pred_sent)
    tag_ents = extract_entities(sent, tag_sent)

    pos_overlap = positive_overlap(pred_ents, train_entities)

    for ent in entities:
        for split in splits:
            scores[ent][split]["tp"] = len(
                [e for e in pred_ents if e[1] == ent and e in tag_ents and overlap_sent[e[2][0]] == split])
            scores[ent][split]["fn"] = len(
                [e for e in tag_ents if e[1] == ent and e not in pred_ents and overlap_sent[e[2][0]] == split])
            scores[ent][split]["fp"] = len([e for i, e in enumerate(pred_ents) if e[1] == ent and e not in tag_ents and
                                            pos_overlap[i] == split])
    return scores


def compute_score_overlap(sents, preds, tags, overlap, train_entities, scheme="iobes", correct=False):
    """Evaluate NER tag predictions.

    Args:
        sents (list) : list of list of words
        preds (list) : list of list of prediction strings
        tags (list) : list of list of ground truth strings
        overlap (list) : list of list of overlap strings ("NEW", "PARTIAL", "EXACT")
        entities (list) : list of possible entity types (default : CoNLL03 entity types)
        scheme (str) : preds and tags tagging scheme (default : "iob")

    Returns:
        dict : per entity and per seen/partially/unseen split TP, FP, FN, precision, recall, f1
    """
    if scheme == "iobes":
        preds = [iobes2iob(s) for s in preds]
        tags = [iobes2iob(s) for s in tags]

    if correct:
        preds = correct_iob(preds)

    splits = ["NEW", "PARTIAL", "EXACT"]
    entities = list(train_entities.keys())

    scores = {ent: {split: {"tp": 0, "fp": 0, "fn": 0} for split in splits} for ent in entities}

    # Count True Positives / False Positives / False Negative
    n_tokens = sum([len(tag) for tag in tags])
    n_phrases = sum([len([t for t in tag if t[0] in ["B"]]) for tag in tags])
    n_found = sum([len([t for t in tag if t[0] in ["B"]]) for tag in preds])

    for sent, pred_sent, tag_sent, overlap_sent in zip(sents, preds, tags, overlap):
        add_scores = sent_eval_overlap(sent, pred_sent, tag_sent, overlap_sent, train_entities)
        for ent in entities:
            for split in splits:
                for k in scores[ent][split].keys():
                    scores[ent][split][k] += add_scores[ent][split][k]

    # print summary
    scores["ALL"] = {split: {"tp": 0, "fp": 0, "fn": 0} for split in splits}
    for split in splits:
        for k in scores["ALL"][split].keys():
            scores["ALL"][split][k] = sum([scores[ent][split][k] for ent in entities])

    # Compute per entity Precision / Recall / F1
    for ent in ["ALL"] + entities:
        # Separate measures (exact / partial / unseen)
        for split in splits:
            if not scores[ent][split]["tp"] == 0:
                scores[ent][split]["r"] = 100 * scores[ent][split]["tp"] / (
                        scores[ent][split]["fn"] + scores[ent][split]["tp"])
                scores[ent][split]["p"] = 100 * scores[ent][split]["tp"] / (
                        scores[ent][split]["fp"] + scores[ent][split]["tp"])
                scores[ent][split]["f1"] = 2 * scores[ent][split]["r"] * scores[ent][split]["p"] / (
                        scores[ent][split]["r"] + scores[ent][split]["p"])

            else:
                scores[ent][split]["p"] = scores[ent][split]["r"] = scores[ent][split]["f1"] = 0

        # Global measures
        for metric in ["tp", "fp", "fn"]:
            scores[ent].update({metric: sum([scores[ent][split][metric] for split in splits])})

        if not scores[ent]["tp"] == 0:
            scores[ent]["p"] = 100 * scores[ent]["tp"] / (scores[ent]["tp"] + scores[ent]["fp"])
            scores[ent]["r"] = 100 * scores[ent]["tp"] / (scores[ent]["tp"] + scores[ent]["fn"])

            scores[ent]["f1"] = 2 * scores[ent]["p"] * scores[ent]["r"] / (scores[ent]["p"] + scores[ent]["r"])

        else:
            scores[ent]["p"] = scores[ent]["r"] = scores[ent]["f1"] = 0

    logging.info(
        "processed {} tokens with {} phrases; found: {} phrases; correct: {}.".format(n_tokens, n_phrases, n_found,
                                                                                      scores["ALL"]["tp"]))

    for ent in ["ALL"] + entities:
        logging.info(
            "{}: \tprecision: {:.2f}; \trecall: {:.2f}; \tf1: {:.2f};\tr_exact: {:.2f};\tr_partial: {:.2f};\tr_new: {:.2f}\tp_exact: {:.2f};\tp_partial: {:.2f};\tp_new: {:.2f};\tf1_exact: {:.2f};\tf1_partial: {:.2f};\tf1_new: {:.2f};".format(
                ent, scores[ent]["p"], scores[ent]["r"], scores[ent]["f1"],
                scores[ent]["EXACT"]["r"], scores[ent]["PARTIAL"]["r"], scores[ent]["NEW"]["r"],
                scores[ent]["EXACT"]["p"], scores[ent]["PARTIAL"]["p"], scores[ent]["NEW"]["p"],
                scores[ent]["EXACT"]["f1"], scores[ent]["PARTIAL"]["f1"], scores[ent]["NEW"]["f1"]))

    return scores


def add_score_overlap(writer, score, n_iter, task="ner"):
    """Add score overlap results to Tensorboard writer."""
    writer.add_scalars("{}/f1".format(task), {ent: score[ent]["f1"] for ent in score.keys()}, n_iter)
    writer.add_scalars("{}/p".format(task), {ent: score[ent]["p"] for ent in score.keys()}, n_iter)
    writer.add_scalars("{}/r".format(task), {ent: score[ent]["r"] for ent in score.keys()}, n_iter)

    for split in ["NEW", "PARTIAL", "EXACT"]:
        writer.add_scalars("{}/r".format(task),
                           {"{}/{}".format(ent, split): score[ent][split]["r"] for ent in score.keys()},
                           n_iter)


def reformat_scores_overlap(scores, model, entities=["ORG", "LOC", "PER", "MISC"]):
    """Reformat scores to append in a pd.DataFrame."""
    reformated = OrderedDict()
    reformated["model"] = model
    for ent in entities + ["ALL"]:
        reformated["Precision {}".format(ent)] = scores[ent]["p"]
        reformated["F1 {}".format(ent)] = scores[ent]["f1"]
        reformated["Recall {}".format(ent)] = scores[ent]["r"]
        reformated["Precision Exact {}".format(ent)] = scores[ent]["EXACT"]["p"]
        reformated["Precision Partial {}".format(ent)] = scores[ent]["PARTIAL"]["p"]
        reformated["Precision New {}".format(ent)] = scores[ent]["NEW"]["p"]
        reformated["F1 Exact {}".format(ent)] = scores[ent]["EXACT"]["f1"]
        reformated["F1 Partial {}".format(ent)] = scores[ent]["PARTIAL"]["f1"]
        reformated["F1 New {}".format(ent)] = scores[ent]["NEW"]["f1"]
        reformated["Recall Exact {}".format(ent)] = scores[ent]["EXACT"]["r"]
        reformated["Recall Partial {}".format(ent)] = scores[ent]["PARTIAL"]["r"]
        reformated["Recall New {}".format(ent)] = scores[ent]["NEW"]["r"]

    return reformated


def reformat_scores(scores, model, entities=["ORG", "LOC", "PER", "MISC"]):
    """Reformat scores to append in a pd.DataFrame."""
    reformated = OrderedDict()
    reformated["model"] = model
    for ent in entities + ["ALL"]:
        reformated["Precision {}".format(ent)] = scores[ent]["p"]
        reformated["F1 {}".format(ent)] = scores[ent]["f1"]
        reformated["Recall {}".format(ent)] = scores[ent]["r"]

    return reformated
