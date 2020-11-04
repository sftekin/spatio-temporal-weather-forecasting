import torch
import torch.nn as nn
from torch.autograd import Variable


class FConvLSTMCell(nn.Module):
    def __init__(self, input_size, input_dim, hidden_dim, flow_dim,
                 kernel_size, bias, device):
        """
        :param tuple of int input_size: width(M) and height(N) of input grid
        :param int input_dim: number of channels (D) of input grid
        :param int hidden_dim: number of channels of hidden state
        :param int kernel_size: size of the convolution kernel
        :param bool bias: weather or not to add the bias
        :param str device: can be 'gpu' and 'cpu'
        """
        super(FConvLSTMCell, self).__init__()
        self.height, self.width = input_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.flow_dim = flow_dim

        self.kernel_size = kernel_size
        self.bias = bias
        self.padding = kernel_size // 2

        self.device = device

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

        self.f_conv = nn.Conv2d(in_channels=self.flow_dim,
                                out_channels=self.hidden_dim,
                                kernel_size=self.kernel_size,
                                padding=self.padding,
                                bias=self.bias)

    def forward(self, input_tensor, cur_state, flow_tensor):

        h_cur, c_cur = cur_state

        flow_out = self.f_conv(flow_tensor)

        h_cur = h_cur * torch.sigmoid(flow_out)

        combined = torch.cat([input_tensor, h_cur], dim=1)

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)

        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        g = torch.tanh(cc_g)
        c_next = f * c_cur + i * g

        o = torch.sigmoid(cc_o)
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size):
        # Create two new tensors with sizes n_layers x batch_size x n_hidden,
        # initialized to zero, for hidden state and cell state of LSTM

        hidden = (Variable(torch.zeros(batch_size,
                                       self.hidden_dim,
                                       self.height,
                                       self.width)).to(self.device),
                  Variable(torch.zeros(batch_size,
                                       self.hidden_dim,
                                       self.height,
                                       self.width)).to(self.device))

        return hidden


