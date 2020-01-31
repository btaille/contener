import copy
import logging

import numpy as np
import torch
from pytorch_pretrained_bert import BertTokenizer
from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertEmbeddings, BertPooler, \
    BertEncoder, BertAttention, BertOutput, BertIntermediate, BertSelfAttention, BertSelfOutput
from torch import nn

from data.data_iterator import pad


def detokenize(sent):
    return " ".join([w for w in sent if not w == '<PAD>'])


def compute_mask(original_tokens, new_tokens):
    mask = [0] * len(new_tokens)

    alignment = []

    i, j = 0, 0
    current_token = ""

    while i < len(original_tokens):
        if new_tokens[j] == "[UNK]":
            logging.info("[UNK] in BERT tokenize")
            return [1] * len(original_tokens), [i for i in range(len(original_tokens))]

        if current_token == "":
            current_m = j

        if new_tokens[j][:2] == "##":
            current_token += new_tokens[j][2:]
        else:
            current_token += new_tokens[j]

        if original_tokens[i] == current_token:
            mask[current_m] = 1
            alignment.append(current_m)
            i += 1
            current_token = ""

        j += 1

    return mask, alignment


def expand_tags(tags, mask, expand_id=0):
    assert len(tags) == sum(mask)

    expanded = [expand_id] * len(mask)

    k = 0
    for i, m in enumerate(mask):
        if m:
            expanded[i] = tags[k]
            k += 1

    return expanded


def bert_tokenize(batch, tokenizer):
    new_words = []
    new_tags = []
    nwords = []
    masks = []
    alignments = []

    for i in range(len(batch["nwords"])):
        text = detokenize(batch["sents"][i])
        tokenized_text = tokenizer.tokenize(text)
        #         mask = [int(token[:2] != "##") for token in tokenized_text]
        mask, alignment = compute_mask(batch["sents"][i][:batch["nwords"][i]], tokenized_text)

        if "tags" in batch.keys():
            tags = batch["tags"][i][:batch["nwords"][i]].cpu().data.numpy()
            tags = expand_tags(tags, mask, expand_id=1)
            tags = [0] + tags + [0]
            new_tags.append(tags)

        tokenized_text = ["[CLS]"] + tokenized_text + ["[SEP]"]
        mask = [0] + mask + [0]
        alignment = [i + 1 for i in alignment]

        new_words.append(tokenizer.convert_tokens_to_ids(tokenized_text))
        nwords.append(len(tokenized_text))
        masks.append(mask)
        alignments.append(alignment)

    new_words = np.array(pad(new_words, 0))
    nwords = np.array(nwords)
    masks = np.array(pad(masks, 0))
    alignments = np.array(pad(alignments, 0))

    output = {"words": batch["words"], "nwords": batch["nwords"],
              "bert_words": new_words, "bert_nwords": nwords,
              "masks": masks, "alignments": alignments}

    if "tags" in batch.keys():
        new_tags = np.array(pad(new_tags, 0))
        output.update({"tags": batch["tags"], "bert_tags": new_tags})

    return output


class CustomBertModel(BertPreTrainedModel):
    """BERT model ("Bidirectional Embedding Representations from a Transformer").

    Params:
        config: a BertConfig class instance with the configuration to build a new model

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `output_all_encoded_layers`: boolean which controls the content of the `encoded_layers` output as described below. Default: `True`.

    Outputs: Tuple of (encoded_layers, pooled_output)
        `encoded_layers`: controled by `output_all_encoded_layers` argument:
            - `output_all_encoded_layers=True`: outputs a list of the full sequences of encoded-hidden-states at the end
                of each attention block (i.e. 12 full sequences for BERT-base, 24 for BERT-large), each
                encoded-hidden-state is a torch.FloatTensor of size [batch_size, sequence_length, hidden_size],
            - `output_all_encoded_layers=False`: outputs only the full sequence of hidden-states corresponding
                to the last attention block of shape [batch_size, sequence_length, hidden_size],
        `pooled_output`: a torch.FloatTensor of size [batch_size, hidden_size] which is the output of a
            classifier pretrained on top of the hidden state associated to the first character of the
            input (`CLS`) to train on the Next-Sentence task (see BERT's paper).

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = modeling.BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = modeling.BertModel(config=config)
    all_encoder_layers, pooled_output = model(input_ids, token_type_ids, input_mask)
    ```
    """

    def __init__(self, config):
        super(CustomBertModel, self).__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, output_all_encoded_layers=True):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids, token_type_ids)
        encoded_layers = self.encoder(embedding_output,
                                      extended_attention_mask)
        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return encoded_layers, pooled_output


class BertEmbedder(nn.Module):
    def __init__(self, bert_model="bert-base-cased", do_lower_case=False, finetune=False, device="cuda"):
        super(BertEmbedder, self).__init__()
        self.device = device

        # Load pretrained Bert Model
        self.tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=do_lower_case)
        self.bert_encoder = CustomBertModel.from_pretrained(bert_model)
        self.finetune = finetune

        self.bert_encoder.to(self.device)

        embeds = self.forward({"sents": [["Hello", "world!"]], "nwords": [2], "words": [0, 1]})

        self.w_embed_dim = embeds["embeddings"].size(-1)
        self.name = bert_model

    def forward(self, data, keys=["all", "embeddings", "cls", "attention"]):
        # WordPiece Tokenization and embedding
        bert_data = bert_tokenize(data, self.tokenizer)

        if self.finetune:
            w_embeds, _ = self.bert_encoder(
                torch.tensor(bert_data["bert_words"]).to(self.device), output_all_encoded_layers=False)

        else:
            self.bert_encoder.eval()
            with torch.no_grad():
                w_embeds, _ = self.bert_encoder(
                    torch.tensor(bert_data["bert_words"]).to(self.device), output_all_encoded_layers=True)
        # Alignment with original tokenization
        alignments = torch.Tensor(bert_data["alignments"]).long().to(self.device)
        indices = alignments.unsqueeze(-1).repeat(1, 1, w_embeds[-1].size(-1))

        # Only return first WordPiece token embedding for each original token
        all_layers = [w.gather(1, indices) for w in w_embeds]

        if self.finetune:
            embeddings = all_layers[-1]

        # Feature-based version of BERT with last four layers
        else:
            embeddings = torch.cat(all_layers[-4:], dim=-1)

        output = {}

        if "all" in keys:
            output.update({"all": all_layers})
        if "embeddings" in keys:
            output.update({"embeddings": embeddings})

        return output
