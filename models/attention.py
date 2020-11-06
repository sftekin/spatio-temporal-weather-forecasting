import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, input_size, hidden_size, input_dim, hidden_dim, attn_dim):
        super(Attention, self).__init__()

        self.hid_conv = nn.Conv2d(in_channels=2*hidden_dim,
                                  out_channels=1,
                                  kernel_size=3,
                                  padding=1)

        self.in_conv = nn.Conv2d(in_channels=input_dim,
                                 out_channels=1,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1)

        self.w = nn.Linear(hidden_size[0]*hidden_size[1], attn_dim)
        self.u = nn.Linear(input_size[0]*input_size[1], attn_dim)
        self.v = nn.Parameter(torch.rand(attn_dim), requires_grad=True)

    def forward(self, input_tensor, hidden):
        """

        :param tuple of torch.tensor hidden: ((B, hidden, M, N), (B, hidden, M, N))
        :param torch.tensor input_tensor: (B, T, m, n, D)
        :return:
        """
        hid_conv_out = self.conv(torch.cat((hidden[0], hidden[1]), dim=1))

        # hidden_vec: (B, 1, M*N)
        hidden_vec = torch.flatten(hid_conv_out, start_dim=2)

        attn_energies = []
        for i in range(input_tensor.shape[-1]):
            # in_vec: (B, 1, m*n)
            in_conv_out = self.in_conv(input_tensor[..., -1])
            in_vec = torch.flatten(in_conv_out, start_dim=2)

            # u(in_vec): (B, 1, attn_dim), w(in_vec): (B, 1, attn_dim), energy: (1)
            energy = torch.sum(self.v * (self.w(hidden_vec) + self.u(in_vec)).tanh())
            attn_energies.append(energy)

        attn_energies = torch.cat(attn_energies, dim=1)
        output = F.softmax(attn_energies, dim=1)

        return output
