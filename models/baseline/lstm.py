import torch
import torch.nn as nn
from torch.autograd import Variable


class LSTMModel(nn.Module):
    def __init__(self, input_size, window_in, window_out, num_layers, selected_dim, hidden_dim, dropout, bias, device):
        super(LSTMModel, self).__init__()
        self.device = device
        self.height, self.width = input_size
        self.input_dim = self.height * self.width
        self.selected_dim = selected_dim
        self.window_in = window_in
        self.window_out = window_out
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.bias = bias
        self.device = device

        self.lstm = nn.LSTM(input_size=self.input_dim,
                            hidden_size=self.hidden_dim,
                            num_layers=self.num_layers,
                            bias=self.bias,
                            dropout=self.dropout,
                            batch_first=True)

        self.linear = nn.Linear(in_features=self.hidden_dim, out_features=self.input_dim)

        self.is_trainable = True

    def forward(self, x, hidden):
        """
        :param x: 5-D tensor of shape (b, t, d, m, n)
        :return: (b, t, d, m, n)
        """
        # (b, t, m, n)
        x_d = x[:, :, self.selected_dim]

        batch_size, win_len = x_d.shape[:2]
        x_d = x_d.reshape((batch_size, win_len, self.input_dim))

        output, (h, c) = self.lstm(x_d, hidden)
        output = output[:, -self.window_out:]
        output = self.linear(output).reshape((batch_size, self.window_out, self.height, self.width))

        # (b, win_out, 1, m, n)
        output = output.unsqueeze(dim=2)

        return output
