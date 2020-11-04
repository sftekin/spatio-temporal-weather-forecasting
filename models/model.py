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

        input_cnn = InputCNN(in_channels=num_series)

        encoder = FConvLSTMCell(input_size=(self.height, self.width),
                               input_dim=self.num_series,
                               hidden_dim=self.encoder_params['hidden_dim'],
                               kernel_size=self.encoder_params['kernel_size'],
                               bias=self.encoder_params['bias'],
                               peephole_con=self.encoder_params['peephole_con'],
                               device=self.device)

        input_attn = Attention(input_dim=1024, hidden_dim=self.num_series, attn_dim=input_attn_dim)

        decoder = FConvLSTMCell(input_size=(self.height, self.width),
                               input_dim=1,
                               hidden_dim=self.decoder_params['hidden_dim'],
                               kernel_size=self.decoder_params['kernel_size'],
                               bias=self.decoder_params['bias'],
                               peephole_con=self.decoder_params['peephole_con'],
                               device=self.device)

        output_attn = Attention(input_dim=self.encoder_params['hidden_dim'])

