import torch
import torch.nn as nn
import torch.nn.functional as F

from models.attention import Attention
from models.f_conv_lstm import FConvLSTMCell
from models.input_cnn import InputCNN


class WeatherModel(nn.Module):
    def __init__(self, window_in, window_out, input_size, num_series,
                 input_attn_dim, temporal_attn_dim, encoder_params, decoder_params, device):
        super().__init__()

        self.height, self.width = input_size
        self.window_in = window_in
        self.window_out = window_out
        self.num_series = num_series

        self.encoder_params = encoder_params
        self.decoder_params = decoder_params
        self.device = device

        self.input_cnn = InputCNN(in_channels=self.window_in)

        self.encoder = FConvLSTMCell(input_size=(self.height, self.width),
                                     input_dim=self.num_series,
                                     hidden_dim=self.encoder_params['hidden_dim'],
                                     flow_dim=self.encoder_params['flow_dim'],
                                     kernel_size=self.encoder_params['kernel_size'],
                                     bias=self.encoder_params['bias'],
                                     padding=self.encoder_params['padding'],
                                     device=self.device)

        self.input_attn = Attention(input_size=(15, 30),
                                    hidden_size=(self.height, self.width),
                                    input_dim=256,
                                    hidden_dim=self.encoder_params['hidden_dim'],
                                    attn_dim=input_attn_dim)

        self.decoder = FConvLSTMCell(input_size=(self.height, self.width),
                                     input_dim=1,
                                     hidden_dim=self.encoder_params['hidden_dim'],
                                     flow_dim=self.decoder_params['flow_dim'],
                                     kernel_size=self.decoder_params['kernel_size'],
                                     bias=self.decoder_params['bias'],
                                     padding=self.decoder_params['padding'],
                                     device=self.device)

        self.out_conv = nn.Conv2d(in_channels=self.encoder_params['hidden_dim'],
                                  out_channels=1,
                                  kernel_size=3,
                                  padding=1,
                                  bias=False)


    def init_hidden(self, batch_size):
        hidden = self.encoder.init_hidden(batch_size)
        return hidden

    def forward(self, x, y, f_x, f_y, hidden):
        """

        :param x: (b, t, d, m, n)
        :type x:
        :param y: (b, t, d, m, n)
        :type y: 
        :param f_x: (b, t, 4, m, n)
        :type f_x:
        :param f_y: (b, t, 4, m, n)
        :type f_y:
        :param hidden: [(b, d', m, n), (b, d', m, n)]
        :type hidden:
        :return:
        :rtype:
        """
        batch_size, win_len, dim_len, height, width = x.shape

        # calculate input attention
        alpha_list = []
        for k in range(dim_len):
            # dim(x_k): (b, 256, m', n')
            x_k = self.input_cnn(x[:, :, k])

            # dim(alpha): (B, 1)
            alpha = self.input_attn(x_k, hidden)
            alpha_list.append(alpha)

        # dim(alpha_tensor): (B, D)
        alpha_tensor = torch.cat(alpha_list, dim=1).squeeze()
        alpha_tensor = F.softmax(alpha_tensor, dim=1)

        # calculate encoder output
        en_out_h, en_out_c = (0, 0)
        for t in range(self.window_in):
            x_t = x[:, t].view(batch_size, dim_len, -1)
            x_tilda = x_t * alpha_tensor.unsqueeze(2)
            x_tilda = x_tilda.view(batch_size, dim_len, height, width)

            hidden = self.encoder(input_tensor=x_tilda,
                                  cur_state=hidden,
                                  flow_tensor=f_x[:, t])

            en_out_h += hidden[0]
            en_out_c += hidden[1]

        de_hidden = (en_out_h, en_out_c)
        de_out = []
        # parse decoder layer and get outputs recursively
        for t in range(self.window_out):
            de_hidden = self.decoder(y[:, t], de_hidden, f_y[:, t])
            conv_out = self.out_conv(de_hidden[0])
            de_out.append(conv_out)

        de_out = torch.stack(de_out, dim=1)

        return de_out
