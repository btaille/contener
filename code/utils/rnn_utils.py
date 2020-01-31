import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class PackedRNN(nn.Module):
    """Wrapper for torch.nn.RNN to feed unordered and unpacked inputs"""

    def __init__(self, rnn):
        super(PackedRNN, self).__init__()
        self.rnn = rnn
        self.batch_first = self.rnn.batch_first

    def forward(self, inputs, seqlens, hidden=None):
        # Sort inputs and lengths by decreasing length
        indices = np.array(np.argsort(seqlens)[::-1])
        seqlens = seqlens[indices]
        inputs = inputs[indices]

        # Pack input sequence, apply RNN and Unpack output
        packed_inputs = pack_padded_sequence(inputs, seqlens, batch_first=self.batch_first)

        self.rnn.flatten_parameters()
        if hidden is None:
            packed_output, hidden = self.rnn(packed_inputs)
        else:
            packed_output, hidden = self.rnn(packed_inputs, hidden)

        output, _ = pad_packed_sequence(packed_output, batch_first=self.batch_first)

        # Reorder outputs and hidden state
        indices_reorder = np.argsort(indices)

        if self.batch_first:
            output = output[indices_reorder]
        else:
            output = output[:, indices_reorder]

        if isinstance(hidden, tuple):
            hidden = hidden[0][:, indices_reorder, :], hidden[1][:, indices_reorder, :]
        else:
            hidden = hidden[:, indices_reorder, :]

        return output, hidden


class ShiftedBiRNN(nn.Module):
    """Given a BiRNN, shift forward RNN outputs to the right and backward RNN outputs to the left in order to
    have a representation of current input based only on its context"""

    def __init__(self, birnn):
        super(ShiftedBiRNN, self).__init__()
        assert birnn.bidirectional
        self.packed_rnn = PackedRNN(birnn)
        self.batch_first = self.packed_rnn.batch_first

    def forward(self, inputs, seqlens, hidden):
        # Use packed RNN
        output, hidden = self.packed_rnn(inputs, seqlens, hidden)

        if not self.batch_first:
            output.transpose(0, 1)

        batch_size = inputs.size(0)
        hidden_dim = hidden[0].size(-1)
        # Separate forward and backward layers
        padded = output.contiguous().view(batch_size, seqlens.max(), 2, hidden_dim)
        # Add zeros at begining and end of sequence
        zeros = torch.zeros(batch_size, 1, 2, hidden_dim).to(inputs.device)
        padded = torch.cat((zeros, padded, zeros), dim=1)
        # Shift forward to the right and backward to the left
        shifted = torch.cat((padded[:, :-2, 0, :], padded[:, 2:, 1, :]), dim=-1)

        if not self.batch_first:
            shifted.transpose(0, 1)

        return shifted, output
