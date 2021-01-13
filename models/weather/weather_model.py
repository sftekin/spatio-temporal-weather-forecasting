import torch
import torch.nn as nn

import torch.nn.functional as F
from models.weather.attention import Attention
from models.baseline.convlstm import ConvLSTMCell


class WeatherModel(nn.Module):

    def __init__(self, input_size, window_in, window_out, num_layers, selected_dim,
                 encoder_params, decoder_params, input_attn_params, temporal_attn_params, device):
        nn.Module.__init__(self)

        self.device = device
        self.input_size = input_size
        self.height, self.width = self.input_size

        self.window_in = window_in
        self.window_out = window_out
        self.num_layers = num_layers
        self.encoder_params = encoder_params
        self.input_attn_params = input_attn_params
        self.temporal_attn_params = temporal_attn_params
        self.decoder_params = decoder_params
        self.selected_dim = selected_dim

        # self.input_cnn = InputCNN(in_channels=self.window_in)

        self.input_attn = Attention(input_dim=input_attn_params["input_dim"],
                                    hidden_dim=input_attn_params["hidden_dim"])

        self.temporal_attn = Attention(input_dim=temporal_attn_params["input_dim"],
                                       hidden_dim=temporal_attn_params["hidden_dim"])

        # define encoder
        self.encoder = self.__define_block(encoder_params)

        # define decoder
        self.decoder = self.__define_block(decoder_params)

        self.hidden = None
        self.is_trainable = True

    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.encoder[i].init_hidden(batch_size))
        return init_states

    def __define_block(self, block_params):
        input_dim = block_params['input_dim']
        hidden_dims = block_params['hidden_dims']
        kernel_size = block_params['kernel_size']
        bias = block_params['bias']
        peephole_con = block_params['peephole_con']

        # Defining block
        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = input_dim if i == 0 else hidden_dims[i - 1]
            cell_list += [self.__create_cell_unit(cur_input_dim,
                                                  hidden_dims[i],
                                                  kernel_size[i],
                                                  bias,
                                                  peephole_con)]
        block = nn.ModuleList(cell_list)

        return block

    def __create_cell_unit(self, cur_input_dim, hidden_dim, kernel_size, bias, peephole_con):
        cell_unit = ConvLSTMCell(input_size=(self.height, self.width),
                                 input_dim=cur_input_dim,
                                 hidden_dim=hidden_dim,
                                 kernel_size=kernel_size,
                                 bias=bias,
                                 device=self.device,
                                 peephole_con=peephole_con)
        return cell_unit

    def forward(self, x, hidden, **kwargs):
        """
        :param input_tensor: 5-D tensor of shape (b, t, m, n, d)
        :param hidden:
        :return: (b, t, m, n, d)
        """
        # forward encoder
        _, cur_states = self.__forward_encoder(x, hidden)

        # reverse the state list
        cur_states = [cur_states[i - 1] for i in range(len(cur_states), 0, -1)]

        # forward decoder block
        dec_output = self.__forward_decoder(x[:, [-1], self.selected_dim], cur_states)

        return dec_output

    def __forward_encoder(self, x, hidden):
        layer_output_list = []
        layer_state_list = []
        b, seq_len, dim_len, height, width = x.shape

        for layer_idx in range(self.num_layers):
            h, c = hidden[layer_idx]
            h_inner = []
            c_inner = []
            for t in range(seq_len):

                if layer_idx == 0:
                    x = self.__forward_input_attn(x, hidden=(h, c))

                h, c = self.encoder[layer_idx](input_tensor=x[:, t, :, :, :],
                                               cur_state=[h, c])
                c_inner.append(c)
                h_inner.append(h)

            layer_h = torch.stack(h_inner, dim=1)
            layer_c = torch.stack(c_inner, dim=1)
            x = layer_h

            layer_output_list.append(layer_h)
            layer_state_list.append([layer_h, layer_c])

        return layer_output_list, layer_state_list

    def __forward_decoder(self, y_t, hidden):
        y_pre = []
        y_next = y_t
        for t in range(self.window_out):
            for layer_idx in range(self.num_layers):
                h, c = hidden[layer_idx]

                if layer_idx == 0:
                    h_cur, c_cur = self.__forward_temporal_attn(y_next, hidden=(h, c))
                else:
                    h_cur = h[:, -1]
                    c_cur = c[:, -1]

                h_cur, c_cur = self.decoder[layer_idx](input_tensor=y_next,
                                                       cur_state=[h_cur, c_cur])
                y_next = h_cur

                h = torch.cat([h, h_cur.unsqueeze(1)], dim=1)
                h = h[:, 1:]

                c = torch.cat([c, c_cur.unsqueeze(1)], dim=1)
                c = c[:, 1:]

                hidden[layer_idx] = (h, c)

            y_pre.append(y_next)

        y_pre = torch.stack(y_pre, dim=1)

        return y_pre

    def __forward_input_attn(self, x, hidden):
        d_dim = x.shape[2]

        # calculate input attention
        alpha_list = []
        for k in range(d_dim):
            # dim(x_k): (b, 256, m', n')
            x_k = x[:, :, k]

            # dim(alpha): (B, 1)
            alpha = self.input_attn(x_k, hidden)
            alpha_list.append(alpha)

        # dim(alpha_tensor): (B, D)
        alpha_tensor = torch.cat(alpha_list, dim=1)
        alpha_tensor = F.softmax(alpha_tensor, dim=1)
        x_tilda = x * alpha_tensor.unsqueeze(1)

        return x_tilda

    def __forward_temporal_attn(self, y_next, hidden):
        h, c = hidden
        seq_len = h.shape[1]

        beta_list = []
        for t in range(seq_len):
            hid_t = (h[:, t], c[:, t])

            beta = self.temporal_attn(y_next, hid_t)
            beta_list.append(beta)

        beta_tensor = torch.cat(beta_list, dim=1)
        beta_tensor = F.softmax(beta_tensor, dim=1)
        h_tilda = torch.sum(h * beta_tensor.unsqueeze(2), dim=1)

        return h_tilda, c[:, -1]
