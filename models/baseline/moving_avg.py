import torch
import torch.nn as nn
from scipy import signal


class MovingAvg(nn.Module):
    def __init__(self, window_in, window_out, mode, output_dim, device):
        super(MovingAvg, self).__init__()
        self.mode = mode
        self.window_in = window_in
        self.window_out = window_out
        self.output_dim = output_dim
        self.device = device

        if mode == "EMA":
            self.mu = 2 / (self.window_in + 1)
            self.is_trainable = False
        else:
            self.weight = self.__init_weight()
            self.is_trainable = True

    def forward(self, x, **kwargs):
        """

        :param x: (b, t, d, m, n)
        :type x:
        :param kwargs:
        :type kwargs:
        :return:
        :rtype:
        """
        x = x[:, :, self.output_dim]
        output = []
        for step in range(self.window_out):

            if self.mode == "EMA":
                # initial value equal to mean of the batch
                pred = torch.mean(x, dim=1, keepdim=True)
                for t in range(1, self.window_in):
                    pred = self.mu * x[:, [t - 1]] + (1 - self.mu) * pred
            else:
                for t in range(self.window_in):
                    x[:, t] *= self.weight[t]
                # average on time dimension
                pred = torch.mean(x, dim=1, keepdim=True)

            # append prediction to the output
            output.append(pred)

            # append the prediction to the input for the next iteration
            x = torch.cat([x, pred], dim=1)
            x = x[:, 1:]

        output = torch.cat(output, dim=1).unsqueeze(2)

        return output

    def __init_weight(self):
        filter_len = 2 * self.window_in
        window = signal.gaussian(filter_len, std=10)

        # attention to right
        window = window[:self.window_in]
        window = torch.from_numpy(window).float()
        window = nn.Parameter(window, requires_grad=True)

        return window


