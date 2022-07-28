import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, input_dim, hidden_dim, attn_channel, kernel_size):
        super(Attention, self).__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        self.H = nn.Conv2d(in_channels=hidden_dim,
                           out_channels=attn_channel,
                           kernel_size=kernel_size,
                           padding=self.padding,
                           bias=False)

        self.W = nn.Conv2d(in_channels=input_dim,
                           out_channels=attn_channel,
                           kernel_size=kernel_size,
                           padding=self.padding,
                           bias=False)

        self.V = nn.Conv2d(in_channels=attn_channel,
                           out_channels=1,
                           kernel_size=kernel_size,
                           padding=self.padding,
                           bias=False)

    def forward(self, input_tensor, hidden):
        """

        :param torch.Tensor input_tensor: (B, T, m, n)
        :param tuple of torch.Tensor hidden: ((B, hidden, M, N), (B, hidden, M, N))
        :return: attention energies, (B, 1, M, N)
        """
        # comb_hid = torch.cat(hidden, dim=1)
        # dim(hid_conv_out) = (B, attn_dim, m, n)
        hid_conv_out = self.H(hidden[0])

        # dim(in_conv_out) = (B, attn_dim, m, n)
        in_conv_out = self.W(input_tensor)

        # dim(energy) = (B, 1, M, N)
        energy = self.V((hid_conv_out + in_conv_out).tanh())

        return energy
