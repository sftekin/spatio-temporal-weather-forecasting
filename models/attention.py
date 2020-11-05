import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, input_dim, hidden_dim, attn_dim):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.attn_dim = attn_dim
        self.input_dim = input_dim

        self.hid_conv = nn.Conv2d(in_channels=2*self.hidden_dim,
                                  out_channels=1,
                                  kernel_size=3,
                                  padding=1)

        self.in_conv = nn.Conv2d(in_channels=self.input_dim,
                                 out_channels=1,
                                 kernel_size=3,
                                 padding=1)

        self.w = nn.Linear(attn_dim, attn_dim)
        self.u = nn.Linear(input_dim, attn_dim)
        self.v = nn.Parameter(torch.rand(hidden_dim), requires_grad=True)

    def forward(self, input_tensor, hidden):
        """

        :param tuple of torch.tensor hidden: ((B, hidden, M, N), (B, hidden, M, N))
        :param torch.tensor input_tensor: (B, input_dim, m, n)
        :return:
        """
        hid_conv_out = self.conv(torch.cat((hidden[0], hidden[1]), dim=1))
        in_conv_out = self.in_conv(input_tensor)

        hidden_vec = torch.flatten(hid_conv_out, start_dim=2)
        in_vec = torch.flatten(in_conv_out, start_dim=2)

        attn_energies = torch.sum(self.v * (self.w(hidden_vec) + self.u(in_vec)).tanh())
        output = F.softmax(attn_energies, dim=0)

        return output
