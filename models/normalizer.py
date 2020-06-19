import torch


class Normalizer:
    def __init__(self, method):
        """

        :param str method:
        """
        self.method = method
        self.fit_dispatcher = {
            "minmax": self.__fit_minmax_scale,
            "standardization": self.__fit_standardization
        }
        self.transform_dispatcher = {
            "minmax": self.__inverse_transform_minmax,
            "standardization": self.__inverse_transform_standard
        }
        self.min_tensor = None
        self.max_tensor = None
        self.std_tensor = None
        self.mean_tensor = None

    def fit(self, input_tensor):
        """

        :param torch.Tensor input_tensor: (..., D)
        :return:
        """
        self.fit_dispatcher[self.method](input_tensor)

    def transform(self, input_tensor):
        """

        :param torch.Tensor input_tensor:
        :return:
        """
        return self.transform_dispatcher[self.method](input_tensor)

    def __fit_minmax_scale(self, input_tensor):
        """

        :param torch.Tensor input_tensor:
        :return:
        """
        dim_d = input_tensor.shape[-1]
        input_tensor = input_tensor.reshape(-1, dim_d)
        self.min_tensor = torch.min(input_tensor, dim=0)[0]
        self.max_tensor = torch.max(input_tensor, dim=0)[0]

    def __fit_standardization(self, input_tensor):
        """

        :param torch.Tensor input_tensor:
        :return:
        """
        dim_d = input_tensor.shape[-1]
        input_tensor = input_tensor.reshape(-1, dim_d)
        self.std_tensor = torch.std(input_tensor, dim=0)
        self.mean_tensor = torch.mean(input_tensor, dim=0)

    def __inverse_transform_minmax(self, input_tensor):
        """

        :param torch.Tensor input_tensor:
        :return:
        """
        min_tensor = self.min_tensor.repeat(*input_tensor.shape[:-1], 1)
        max_tensor = self.max_tensor.repeat(*input_tensor.shape[:-1], 1)
        return torch.mul(input_tensor, max_tensor - min_tensor) + min_tensor

    def __inverse_transform_standard(self, input_tensor):
        """

        :param torch.Tensor input_tensor:
        :return:
        """
        std_tensor = self.std_tensor.repeat(*input_tensor.shape[:-1], 1)
        mean_tensor = self.mean_tensor.repeat(*input_tensor.shape[:-1], 1)
        return torch.mul(input_tensor, std_tensor) + mean_tensor
