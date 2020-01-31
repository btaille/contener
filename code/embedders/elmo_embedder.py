from typing import Union, List, Dict

import torch
import torch.nn as nn
from allennlp.modules.elmo import Elmo, batch_to_ids
from allennlp.nn.util import remove_sentence_boundaries

from global_vars import EMBEDDINGS_DIR


class CustomElmo(Elmo):
    def forward(self,  # pylint: disable=arguments-differ
                inputs: torch.Tensor,
                word_inputs: torch.Tensor = None) -> Dict[str, Union[torch.Tensor, List[torch.Tensor]]]:
        """
        Parameters
        ----------
        inputs: ``torch.Tensor``, required.
        Shape ``(batch_size, timesteps, 50)`` of character ids representing the current batch.
        word_inputs : ``torch.Tensor``, required.
            If you passed a cached vocab, you can in addition pass a tensor of shape
            ``(batch_size, timesteps)``, which represent word ids which have been pre-cached.

        Returns
        -------
        Dict with keys:
        ``'elmo_representations'``: ``List[torch.Tensor]``
            A ``num_output_representations`` list of ELMo representations for the input sequence.
            Each representation is shape ``(batch_size, timesteps, embedding_dim)``
        ``'mask'``:  ``torch.Tensor``
            Shape ``(batch_size, timesteps)`` long tensor with sequence mask.
        """
        # reshape the input if needed
        original_shape = inputs.size()
        if len(original_shape) > 3:
            timesteps, num_characters = original_shape[-2:]
            reshaped_inputs = inputs.view(-1, timesteps, num_characters)
        else:
            reshaped_inputs = inputs

        if word_inputs is not None:
            original_word_size = word_inputs.size()
            if self._has_cached_vocab and len(original_word_size) > 2:
                reshaped_word_inputs = word_inputs.view(-1, original_word_size[-1])
            elif not self._has_cached_vocab:
                reshaped_word_inputs = None
            else:
                reshaped_word_inputs = word_inputs
        else:
            reshaped_word_inputs = word_inputs

        # run the biLM
        bilm_output = self._elmo_lstm(reshaped_inputs, reshaped_word_inputs)
        layer_activations = bilm_output['activations']
        mask_with_bos_eos = bilm_output['mask']

        # compute the elmo representations
        representations = []
        for i in range(len(self._scalar_mixes)):
            scalar_mix = getattr(self, 'scalar_mix_{}'.format(i))
            representation_with_bos_eos = scalar_mix(layer_activations, mask_with_bos_eos)
            if self._keep_sentence_boundaries:
                processed_representation = representation_with_bos_eos
                processed_mask = mask_with_bos_eos
            else:
                representation_without_bos_eos, mask_without_bos_eos = remove_sentence_boundaries(
                    representation_with_bos_eos, mask_with_bos_eos)
                processed_representation = representation_without_bos_eos
                processed_mask = mask_without_bos_eos

            representations.append(self._dropout(processed_representation))

        if not self._keep_sentence_boundaries:
            for i, activation in enumerate(layer_activations):
                layer_activations[i], _ = remove_sentence_boundaries(activation, mask_with_bos_eos)

        # reshape if necessary
        if word_inputs is not None and len(original_word_size) > 2:
            mask = processed_mask.view(original_word_size)
            elmo_representations = [representation.view(original_word_size + (-1,))
                                    for representation in representations]

        elif len(original_shape) > 3:
            mask = processed_mask.view(original_shape[:-1])
            elmo_representations = [representation.view(original_shape[:-1] + (-1,))
                                    for representation in representations]
        else:
            mask = processed_mask
            elmo_representations = representations

        return {'embeddings': elmo_representations[0], 'activations': layer_activations, 'mask': mask,
                'elmo_representations': elmo_representations}


class ElmoEmbedder(nn.Module):
    def __init__(self, options_file=EMBEDDINGS_DIR + "elmo/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json",
                 weight_file=EMBEDDINGS_DIR + "elmo/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5",
                 num_output_representations=1, dropout=0., device="cuda"):
        super(ElmoEmbedder, self).__init__()

        self.options_file = options_file
        self.weight_file = weight_file
        self.num_output_representations = num_output_representations
        self.dropout = dropout
        self.device = device

        self.elmo = CustomElmo(self.options_file, self.weight_file,
                               num_output_representations=self.num_output_representations,
                               dropout=self.dropout)
        self.elmo.to(device)
        self.name = "elmo"

        embeds = self.forward({"words": torch.Tensor(1), "sents": [["Hello", "world!"]], "nwords": [2]})
        self.w_embed_dim = embeds["embeddings"].size(-1)

    def forward(self, data):
        char_ids = batch_to_ids(data["sents"]).to(self.device)
        return self.elmo(char_ids)


class ElmoLayerEmbedder(nn.Module):
    def __init__(self, layer, options_file=EMBEDDINGS_DIR + "elmo/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json",
                 weight_file=EMBEDDINGS_DIR + "elmo/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5",
                 num_output_representations=1, dropout=0., device="cuda"):
        super(ElmoLayerEmbedder, self).__init__()

        assert layer < 3
        self.layer = layer
        self.options_file = options_file
        self.weight_file = weight_file
        self.num_output_representations = num_output_representations
        self.dropout = dropout
        self.device = device

        self.elmo = CustomElmo(self.options_file, self.weight_file,
                               num_output_representations=self.num_output_representations,
                               dropout=self.dropout)
        self.elmo.to(device)
        self.name = "elmo"

        embeds = self.forward({"words": torch.Tensor(1), "sents": [["Hello", "world!"]], "nwords": [2]})
        self.w_embed_dim = embeds["embeddings"].size(-1)

    def forward(self, data):
        char_ids = batch_to_ids(data["sents"]).to(self.device)
        return {"embeddings": self.elmo(char_ids)["activations"][self.layer]}
