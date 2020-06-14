import torch


class Normalizer:
    def __init__(self, method):
        self.method = method

        self.fit_dispatcher = {
            "minmax": self.__fit_minmax_scale,
            "standardization": self.__fit_standardization
        }
        self.transform_dispatcher = {
            "minmax": self.__transform_minmax_scale,
            "standardization": self.__transform_standardization
        }

        self.min_tensor = None
        self.max_tensor = None

    def fit(self, input_tensor):
        """

        :param torch.tensor input_tensor: (..., D)
        :return:
        """
        self.norm_dispatcher[self.method](input_tensor)

    def __fit_minmax_scale(self, input_tensor):
        """

        :param torch.Tensor input_tensor:
        :return:
        """
        dim_d = input_tensor.shape[-1]
        input_tensor = input_tensor.reshape(-1, dim_d)
        self.min_tensor = torch.min(input_tensor, dim=0)[0]
        self.max_tensor = torch.max(input_tensor, dim=0)[0]

    def __fit_standardization(self):
        pass

    def transform(self):
        pass

    def __transform_minmax_scale(self):
        pass

    def __transform_standardization(self):
        pass
