import torch
import torch.nn as nn

from models.attention import Attention
from models.f_conv_lstm import FConvLSTMCell
from models.input_cnn import InputCNN


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

        self.input_cnn = InputCNN(in_channels=num_series)

        self.encoder = FConvLSTMCell(input_size=(self.height, self.width),
                                     input_dim=self.num_series,
                                     hidden_dim=self.encoder_params['hidden_dim'],
                                     flow_dim=self.encoder_params['flow_dim'],
                                     kernel_size=self.encoder_params['kernel_size'],
                                     bias=self.encoder_params['bias'],
                                     device=self.device)

        self.input_attn = Attention(input_dim=1024, hidden_dim=self.num_series, attn_dim=input_attn_dim)

        self.decoder = FConvLSTMCell(input_size=(self.height, self.width),
                                     input_dim=1,
                                     hidden_dim=self.decoder_params['hidden_dim'],
                                     flow_dim=self.decoder_params['flow_dim'],
                                     kernel_size=self.decoder_params['kernel_size'],
                                     bias=self.decoder_params['bias'],
                                     device=self.device)

        self.output_attn = Attention(input_dim=self.encoder_params['hidden_dim'])

    def forward(self, input_tensor, flow_tensor, hidden):
        """

        :param input_tensor: (b, t, d, m, n)
        :type input_tensor:
        :param flow_tensor: (b, t, d, m, n)
        :type flow_tensor:
        :param hidden_dim: [(b, d', m, n), (b, d', m, n)]
        :type hidden_dim:
        :return:
        :rtype:
        """
        win_len = input_tensor.shape[1]

        cnn_out = []
        for t in range(win_len):
            # dim(x_t) = (b, 1024, m', n')
            x_t = self.input_cnn(input_tensor[:, t])
            cnn_out.append(x_t)
        cnn_out = torch.stack(cnn_out, dim=1)

        self.input_attn()
