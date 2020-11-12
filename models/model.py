import torch
import torch.nn as nn
import torch.nn.functional as F

from models.attention import Attention
from models.f_conv_lstm import FConvLSTMCell
from models.input_cnn import InputCNN


class WeatherModel(nn.Module):
    def __init__(self, window_length, input_size, encoder_params,
                 decoder_params, input_attn_dim, temporal_attn_dim, device):
        super().__init__()

        self.height, self.width = input_size
        self.win_length = window_length

        self.encoder_params = encoder_params
        self.decoder_params = decoder_params
        self.device = device

        self.input_cnn = InputCNN(in_channels=window_length)

        self.encoder = FConvLSTMCell(input_size=(self.height, self.width),
                                     input_dim=self.num_series,
                                     hidden_dim=self.encoder_params['hidden_dim'],
                                     flow_dim=self.encoder_params['flow_dim'],
                                     kernel_size=self.encoder_params['kernel_size'],
                                     bias=self.encoder_params['bias'],
                                     device=self.device)

        self.input_attn = Attention(input_size=(15, 30),
                                    hidden_size=(self.height, self.width),
                                    input_dim=256,
                                    hidden_dim=self.encoder_params['hidden_dim'],
                                    attn_dim=input_attn_dim)

        self.decoder = FConvLSTMCell(input_size=(self.height, self.width),
                                     input_dim=1,
                                     hidden_dim=self.decoder_params['hidden_dim'],
                                     flow_dim=self.decoder_params['flow_dim'],
                                     kernel_size=self.decoder_params['kernel_size'],
                                     bias=self.decoder_params['bias'],
                                     device=self.device)

        self.output_attn = Attention(input_size=(15, 30),
                                     hidden_size=self.decoder_params['hidden_size'],
                                     input_dim=self.encoder_params['hidden_dim'],
                                     hidden_dim=self.decoder_params['hidden_dim'],
                                     attn_dim=temporal_attn_dim)

    def forward(self, input_tensor, output_tensor, flow_tensor, hidden):
        """

        :param input_tensor: (b, t, d, m, n)
        :type input_tensor:
        :param flow_tensor: (b, t, d, m, n)
        :type flow_tensor:
        :param hidden: [(b, d', m, n), (b, d', m, n)]
        :type hidden:
        :return:
        :rtype:
        """
        batch_size, win_len, dim_len, height, width = input_tensor.shape

        # calculate input attention
        alpha_list = []
        for k in range(dim_len):
            # dim(x_k): (b, 256, m', n')
            x_k = self.input_cnn(input_tensor[:, :, k])

            # dim(alpha): (B, 1)
            alpha = self.input_attn(x_k, hidden)
            alpha_list.append(alpha)

        # dim(alpha_tensor): (B, D)
        alpha_tensor = torch.cat(alpha_list, dim=1).squeeze()
        alpha_tensor = F.softmax(alpha_tensor, dim=1)

        # calculate encoder output
        en_out = []
        for t in range(win_len):
            x_t = input_tensor[:, t].view(batch_size, dim_len, -1)
            x_tilda = x_t * alpha_tensor.unsqueeze(2)
            x_tilda = x_tilda.view(batch_size, dim_len, height, width)

            h, c = self.encoder(input_tensor=x_tilda, cur_state=hidden, flow_tensor=flow_tensor)
            en_out.append(h)

        de_hidden = self.decoder.init_hidden(batch_size)
        de_out = []

        # parse decoder layer and get outputs recursively
        for t in range(self.win_length):

            beta_list = []
            for k in range(len(en_out)):
                beta = self.output_attn(en_out[k], de_hidden)
                beta_list.append(beta)
            # dim(beta_tensor): (B, T)
            beta_tensor = F.softmax(torch.cat(beta_list, dim=1).squeeze())

            context_t = []
            for k in range(len(en_out)):
                en_dim = en_out[k].shape[2]
                h_k = en_out[k].view(batch_size, en_dim, -1)
                context_t.append(h_k * beta_tensor.unsqueeze(2))

            context_t = torch.sum(torch.cat(context_t, 1), 1)
            context_t = context_t.view(batch_size, self.decoder.height, self.decoder.width)

            de_in = torch.cat([context_t, output_tensor], dim=1)
            h, c = self.decoder(de_in, de_hidden)
            de_out.append(h)

        de_out = torch.stack(de_out, dim=1)

        return de_out

