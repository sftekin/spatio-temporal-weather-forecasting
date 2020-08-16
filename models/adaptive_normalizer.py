import torch


class AdaptiveNormalizer:
    def __init__(self, seasonality=''):
        self.seasonality = seasonality

    def norm(self, input_tensor):
        """
        Normalizes input tensor.

        :param torch.tensor input_tensor: (T, M, N, D)
        :return: normalized tensor
        :rtype: torch.Tensor
        """
        last_value = input_tensor[-1] + 1e-6

        if self.seasonality:
            pass

        normalized = torch.log(input_tensor / last_value + 1e-6)
        return normalized

    def inv_norm(self, input_tensor):
        """
        Take inverse normalization of the input tensor.

        :param torch.tensor input_tensor: (T, M, N, 1)
        :return:
        """
        last_value = input_tensor[-1]

        if self.seasonality:
            pass

        inv_normalized = torch.exp(input_tensor) * last_value
        return inv_normalized
