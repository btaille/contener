import torch
import torch.nn as nn

from utils.rnn_utils import PackedRNN


class BiLSTMEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, bidirectional=True, dropout=0., n_layers=1):
        super(BiLSTMEncoder, self).__init__()
        self.drop = nn.Dropout(dropout) if dropout else lambda x: x

        self.hidden_dim = hidden_dim
        self.output_dim = 2 * self.hidden_dim if bidirectional else self.hidden_dim

        self.n_layers = n_layers

        self.layers = [PackedRNN(nn.LSTM(input_dim, hidden_dim, bidirectional=bidirectional, batch_first=True))]

        for _ in range(self.n_layers - 1):
            self.layers.append(
                PackedRNN(nn.LSTM(self.output_dim, hidden_dim, bidirectional=bidirectional, batch_first=True)))

        self.name = "bilstm"

        for i, layer in enumerate(self.layers):
            self.add_module("lstm_{}".format(i), layer)

    def forward(self, data, input_key="embeddings", nwords=None):
        inputs = data[input_key]

        if nwords is None:
            nwords = data["nwords"]

        for l in self.layers:
            inputs = self.drop(inputs)
            inputs, self.hidden = l(inputs, nwords)

        # Final hidden state
        final_hidden = torch.cat([self.hidden[0][0], self.hidden[0][1]], dim=-1)
        final_cell = torch.cat([self.hidden[1][0], self.hidden[1][1]], dim=-1)

        return {"output": inputs, "last_hidden": final_hidden, "last_cell": final_cell}


class LinearEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, activation=nn.ReLU, dropout=0.):
        super(LinearEncoder, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.input_dim = input_dim
        self.output_dim = hidden_dim
        self.activation = activation()
        self.linear = nn.Linear(input_dim, hidden_dim)

    def forward(self, data, input_key):
        inputs = self.drop(data[input_key])
        output = self.linear(inputs)

        if self.activation is not None:
            output = self.activation(output)

        return {"output": output}
