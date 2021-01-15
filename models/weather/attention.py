import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Attention, self).__init__()

        self.H = nn.Conv2d(in_channels=hidden_dim,
                           out_channels=5,
                           kernel_size=3,
                           padding=1,
                           bias=False)

        self.W = nn.Conv2d(in_channels=input_dim,
                           out_channels=5,
                           kernel_size=3,
                           stride=1,
                           padding=1,
                           bias=False)

        self.V = nn.Conv2d(in_channels=5,
                           out_channels=1,
                           kernel_size=3,
                           stride=1,
                           padding=1,
                           bias=False)

    def forward(self, input_tensor, hidden):
        """

        :param torch.tensor input_tensor: (B, T, m, n)
        :param tuple of torch.tensor hidden: ((B, hidden, M, N), (B, hidden, M, N))
        :return: attention energies, (B, 1)
        """
        # comb_hid = torch.cat(hidden, dim=1)

        hid_conv_out = self.H(hidden[0])

        # in_vec: (B, 1, m*n)
        in_conv_out = self.W(input_tensor)

        # u(in_vec): (B, 1, attn_dim), w(in_vec): (B, 1, attn_dim), energy: (B, 1)
        energy = self.V((hid_conv_out + in_conv_out).tanh())

        return energy
