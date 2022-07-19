import torch
from torch import nn


class TrajGRU(nn.Module):
    def __init__(self, input_size, window_in, window_out, encoder_params, decoder_params, device):
        super(TrajGRU, self).__init__()

        self.device = device
        self.input_size = input_size
        self.height, self.width = self.input_size

        self.window_in = window_in
        self.window_out = window_out

        self.encoder_params = encoder_params
        self.decoder_params = decoder_params

        self.encoder = self.__create_cell_unit(encoder_params)
        self.decoder = self.__create_cell_unit(decoder_params)

        self.hidden = None
        self.is_trainable = True

    def init_hidden(self, batch_size):
        init_states = self.encoder.init_hidden(batch_size)
        return init_states

    def __create_cell_unit(self, params):
        cell_unit = TrajGRUCell(input_size=(self.height, self.width),
                                input_dim=params["input_dim"],
                                hidden_dim=params["hidden_dim"],
                                kernel_size=params["kernel_size"],
                                bias=params["bias"],
                                connection=params["connection"],
                                device=self.device)
        return cell_unit

    def forward(self, x, hidden):
        batch_size = x.shape[0]

        cur_states = self.__forward_block(x, hidden, 'encoder')

        decoder_input = torch.zeros((batch_size, self.window_out,
                                     self.decoder_params['input_dim'],
                                     self.height, self.width)).to(self.device)
        dec_output = self.__forward_block(decoder_input, cur_states[:, -1], 'decoder')

        return dec_output

    def __forward_block(self, x, hidden, block_name):
        block = getattr(self, block_name)
        h = hidden
        seq_len = x.size(1)
        output = []
        for t in range(seq_len):
            h = block(x[:, t], h_prev=h)
            output.append(h)
        output = torch.stack(output, dim=1)

        return output


class TrajGRUCell(nn.Module):
    """
    TrajGru Cell
    """

    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, bias, connection, device):
        """
        :param input_size: (int, int) width(M) and height(N) of input grid
        :param input_dim: int, number of channels (D) of input grid
        :param hidden_dim: int, number of channels of hidden state
        :param kernel_size: (int, int) size of the convolution kernel
        :param bias: bool weather or not to add the bias
        """
        super(TrajGRUCell, self).__init__()

        self.height, self.width = input_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.connection = connection
        self.device = device

        self.kernel_size = kernel_size
        self.bias = bias
        self.padding = self.kernel_size // 2

        self.conv_input = nn.Conv2d(in_channels=self.input_dim,
                                    out_channels=3 * self.hidden_dim,
                                    kernel_size=self.kernel_size,
                                    padding=self.padding,
                                    bias=self.bias)

        self.projecting_channels = []
        for _ in range(self.connection):
            self.projecting_channels.append(nn.Conv2d(in_channels=self.hidden_dim,
                                                      out_channels=3 * self.hidden_dim,
                                                      kernel_size=1))

        self.projecting_channels = nn.ModuleList(self.projecting_channels)

        self.sub_net = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                                 out_channels=2 * self.connection,
                                 kernel_size=5,
                                 padding=2)

    def forward(self, x, h_prev):
        """
        :param x: (b, d, m, n)
        :type x: tensor
        :param h_prev: (b, d, m, n)
        :type h_prev: tensor
        :return: (b, d, m, n)
        """
        input_conv = self.conv_input(x)

        x_z, x_r, x_h = torch.split(input_conv, self.hidden_dim, dim=1)

        traj_tensor = None
        for local_link, warped in enumerate(self.__warp(x=x, h=h_prev)):
            if local_link == 0:
                traj_tensor = self.projecting_channels[local_link](warped)
            else:
                traj_tensor += self.projecting_channels[local_link](warped)

        h_z, h_r, h_h = torch.split(traj_tensor, self.hidden_dim, dim=1)

        z = torch.sigmoid(x_z + h_z)
        r = torch.sigmoid(x_r + h_r)
        h = nn.functional.leaky_relu(x_h + r * h_h, negative_slope=0.2)

        h_next = (1 - z) * h + z * h_prev

        return h_next

    def init_hidden(self, batch_size):
        """
        # Create new tensor with sizes n_layers x batch_size x n_hidden,
        # initialized to zero, for hidden state of GRU
        :param device:
        :param batch_size: int
        :return:(b, d, m, n) tensor
        """
        hidden = torch.zeros(batch_size, self.hidden_dim,
                             self.height, self.width, requires_grad=True).to(self.device)

        return hidden

    def __warp(self, x, h):
        """
        :param x: (b, d, m, n)
        :type x: tensor
        :param h: (b, d, m, n)
        :type h: tensor
        :return: yields warped tensor
        """
        combined = torch.cat([x, h], dim=1)
        combined_conv = self.sub_net(combined)

        # (b, 2L, m, n) --> (b, m, n, 2L)
        combined_conv = combined_conv.permute(0, 2, 3, 1)

        # scale to [0, 1]
        combined_conv = (combined_conv - combined_conv.min()) / \
                        (combined_conv.max() - combined_conv.min())
        # scale to [-1, 1]
        combined_conv = 2 * combined_conv - 1

        for l in range(0, self.connection, 2):
            # (b, m, n, 2)
            grid = combined_conv[..., l:l + 2]
            warped = nn.functional.grid_sample(h, grid, mode='bilinear', align_corners=False)

            yield warped
