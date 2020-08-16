import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class ConvLSTMCell(nn.Module):
    def __init__(self, input_size, input_dim, hidden_dim,
                 kernel_size, bias, device, peephole_con=False):
        """
        :param tuple of int input_size: width(M) and height(N) of input grid
        :param int input_dim: number of channels (D) of input grid
        :param int hidden_dim: number of channels of hidden state
        :param int kernel_size: size of the convolution kernel
        :param bool bias: weather or not to add the bias
        :param str device: can be 'gpu' and 'cpu'
        :param bool peephole_con: flag for peephole connections
        """
        super(ConvLSTMCell, self).__init__()
        self.height, self.width = input_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.bias = bias
        self.padding = kernel_size // 2

        self.peephole_con = peephole_con
        self.device = device

        if peephole_con:
            self.w_peep = None

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):

        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)

        if self.peephole_con:
            w_ci, w_cf, w_co = torch.split(self.w_peep, self.hidden_dim, dim=1)
            cc_i += w_ci * c_cur
            cc_f += w_cf * c_cur

        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        g = torch.tanh(cc_g)
        c_next = f * c_cur + i * g

        o = torch.sigmoid(cc_o + w_co * c_next) if self.peephole_con else torch.sigmoid(cc_o)
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size):
        # Create two new tensors with sizes n_layers x batch_size x n_hidden,
        # initialized to zero, for hidden state and cell state of LSTM
        if self.peephole_con:
            self.w_peep = Variable(torch.zeros(batch_size,
                                               self.hidden_dim * 3,
                                               self.height,
                                               self.width)).to(self.device)

        hidden = (Variable(torch.zeros(batch_size,
                                       self.hidden_dim,
                                       self.height,
                                       self.width)).to(self.device),
                  Variable(torch.zeros(batch_size,
                                       self.hidden_dim,
                                       self.height,
                                       self.width)).to(self.device))

        return hidden


class Attention(nn.Module):
    def __init__(self, input_dim, hidden_dim, attn_dim):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.attn_dim = attn_dim

        self.conv = nn.Conv2d(in_channels=2*self.hidden_dim,
                              out_channels=1,
                              kernel_size=3,
                              padding=1)

        self.w = nn.Linear(attn_dim, attn_dim)
        self.u = nn.Linear(input_dim, attn_dim)
        self.v = nn.Parameter(torch.rand(hidden_dim))

    def forward(self, hidden, input_tensor):
        """

        :param tuple of torch.tensor hidden: ((B, hidden, M, N), (B, hidden, M, N))
        :param torch.tensor input_tensor: (B, input_dim)
        :return:
        """
        batch_size = input_tensor.shape[0]

        conv_out = self.conv(torch.cat((hidden[0], hidden[1]), dim=1)).squeeze()
        hidden_vec = conv_out.reshape(batch_size, -1)
        attn_energies = torch.sum(self.v * (self.w(hidden_vec) + self.u(input_tensor)).tanh())
        output = F.softmax(attn_energies, dim=0)

        return output


class InputCNN(nn.Module):

    class DoubleConv(nn.Module):
        def __init__(self, in_channels, out_channels, mid_channels):
            super(InputCNN.DoubleConv, self).__init__()

            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        def forward(self, x):
            return self.double_conv(x)

    class Down(nn.Module):

        def __init__(self, in_channels, out_channels):
            super().__init__()

            self.maxpool_conv = nn.Sequential(
                nn.MaxPool2d(kernel_size=2),
                InputCNN.DoubleConv(in_channels, out_channels)
            )

        def forward(self, x):
            return self.maxpool_conv(x)

    def __init__(self, in_channels, out_channels=64, mid_channels=32):
        super(InputCNN, self).__init__()

        self.n_channels = in_channels
        self.out_channels = out_channels

        self.inc = InputCNN.DoubleConv(in_channels, out_channels, mid_channels)
        self.down1 = InputCNN.Down(64, 128)
        self.down2 = InputCNN.Down(128, 256)
        self.down3 = InputCNN.Down(256, 512)
        self.down4 = InputCNN.Down(512, 1024)


class WeatherModel(nn.Module):
    def __init__(self, num_series, height, width, encoder_params,
                 decoder_params, input_attn_dim, temporal_attn_dim, device):
        super().__init__()

        self.height = height
        self.width = width
        self.num_series = num_series
        self.encoder_params = encoder_params
        self.decoder_params = decoder_params
        self.device = device

        input_cnn = InputCNN(in_channels=num_series)

        encoder = ConvLSTMCell(input_size=(self.height, self.width),
                               input_dim=self.num_series,
                               hidden_dim=self.encoder_params['hidden_dim'],
                               kernel_size=self.encoder_params['kernel_size'],
                               bias=self.encoder_params['bias'],
                               peephole_con=self.encoder_params['peephole_con'],
                               device=self.device)

        input_attn = Attention(input_dim=1024, hidden_dim=self.num_series, attn_dim=input_attn_dim)

        decoder = ConvLSTMCell(input_size=(self.height, self.width),
                               input_dim=1,
                               hidden_dim=self.decoder_params['hidden_dim'],
                               kernel_size=self.decoder_params['kernel_size'],
                               bias=self.decoder_params['bias'],
                               peephole_con=self.decoder_params['peephole_con'],
                               device=self.device)

        output_attn = Attention(input_dim=self.encoder_params['hidden_dim'])

