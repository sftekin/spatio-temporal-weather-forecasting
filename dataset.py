import torch
import numpy as np

from torch.utils.data import Dataset


class WeatherDataset:
    def __init__(self, weather_data, input_dim, output_dim, atm_dim, window_len, batch_size):
        """

        :param input_dim:
        :param output_dim:
        :param window_len:
        :param atm_dim:
        :param batch_size:
        """
        self.weather_data = weather_data
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.atm_dim = atm_dim
        self.window_len = window_len
        self.batch_size = batch_size

    def next(self):
        """
        Iterator function. It yields x and y with the following shapes
        x: (B, T, M, N, self.input_dim)
        y: (B, T, M, N, 1)

        :return: x, y
        :rtype: torch.tensor, torch.tensor
        """
        weather_data = self.__configure_data(in_data=self.weather_data)
        weather_data = self.__create_buffer(in_data=weather_data)

        for i in range(len(weather_data)):
            batch_data = self.__load_batch(batch=weather_data[i])
            batch_data = batch_data[:, :, self.atm_dim]

            # create x and y
            x = torch.tensor(batch_data[..., self.input_dim], dtype=torch.float32)
            y = torch.tensor(batch_data[..., self.output_dim], dtype=torch.float32)

            yield x, y

    def __configure_data(self, in_data):
        """

        :param numpy.ndarray in_data: input path array with the shape of (T,)
        :return: batched array with the shape of  (B, T')
        """
        t_dim = len(in_data)

        # Keep only enough time steps to make full batches
        n_batches = t_dim // (self.batch_size * self.window_len)
        end_time_step = n_batches * self.batch_size * self.window_len
        in_data = in_data[:end_time_step]

        # Reshape into batch_size rows
        in_data = in_data.reshape((self.batch_size, -1))

        return in_data

    def __create_buffer(self, in_data):
        """

        :param numpy.ndarray in_data:
        :return:
        """
        total_frame = in_data.shape[1]

        stacked_data = []
        for i in range(0, total_frame, self.window_len):
            batch = in_data[:, i:i+self.window_len]
            stacked_data.append(batch)

        return stacked_data

    @staticmethod
    def __load_batch(batch):
        """


        :param numpy.ndarray batch:
        :return:
        :rtype: numpy.ndarray
        """
        batch_size, win_len = batch.shape
        flatten_b = batch.flatten()

        list_arr = []
        for i in range(len(flatten_b)):
            list_arr.append(np.load(flatten_b[i]))

        return_batch = np.stack(list_arr, axis=0)
        other_dims = return_batch.shape[1:]
        return_batch = return_batch.reshape((batch_size, win_len, *other_dims))

        return return_batch
