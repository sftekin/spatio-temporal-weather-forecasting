import torch
import numpy as np

from torch.utils.data import Dataset


class WeatherDataset:
    def __init__(self, weather_data, input_dim, output_dim,
                 window_in_len, window_out_len, batch_size, normalizer, shuffle):
        """

        :param input_dim:
        :param output_dim:
        :param window_in_len:
        :param window_out_len:
        :param batch_size:
        :param shuffle:
        """
        self.weather_data = weather_data
        self.input_dim = input_dim
        self.output_dim = output_dim if isinstance(output_dim, list) else [output_dim]
        self.window_in_len = window_in_len
        self.window_out_len = window_out_len
        self.total_window_len = window_in_len + window_out_len
        self.batch_size = batch_size
        self.num_iter = 0
        self.normalizer = normalizer
        self.shuffle = shuffle

    def next(self):
        """
        Iterator function. It yields x and y with the following shapes
        x: (B, T, M, N, self.input_dim)
        y: (B, T, M, N, 1)

        :return: x, y
        :rtype: torch.tensor, torch.tensor
        """
        weather_data = self.__create_buffer(in_data=self.weather_data)
        self.num_iter = len(weather_data)

        prev_batch = None
        for i in range(self.num_iter):
            batch_data = torch.from_numpy(self.__load_batch(batch=weather_data[i]))

            if self.normalizer:
                batch_data = self.normalizer.norm(batch_data)

            # create x and y
            x = batch_data[:, :self.window_in_len, ..., self.input_dim]
            y = batch_data[:, self.window_in_len:, ..., self.output_dim]

            # create flow matrix
            if prev_batch is None:
                prev_batch = torch.zeros_like(batch_data)

            flow_batch = torch.cat([prev_batch[:, [-1]], batch_data], dim=1)  # (B, T+1, M, N, D)
            f_x = self.create_flow_mat(flow_batch)

            yield x, y, f_x

    def __create_buffer(self, in_data):
        """
        Creates the buffer of frames.

        :param numpy.ndarray in_data:
        :return: batches as list
        :rtype: list of numpy.ndarray
        """
        total_frame = len(in_data)

        all_data = []
        batch = []
        j = 0
        for i in range(total_frame-self.total_window_len):
            if j < self.batch_size:
                batch.append(in_data[i:i+self.total_window_len])
                j += 1
            else:
                all_data.append(np.stack(batch, axis=0))
                batch = []
                j = 0

        if self.shuffle:
            all_data = np.stack(all_data)
            all_data = all_data.reshape(len(all_data)*self.batch_size, -1)
            all_data = all_data[np.random.permutation(len(all_data))]
            all_data = all_data.reshape(-1, self.batch_size, all_data.shape[-1])

        return all_data

    def create_flow_mat(self, x):
        """

        :param y: (B, T+1, M, N, D)
        :type y:
        :return:
        :rtype:
        """
        batch_dim, seq_dim, height, width, d_dim = x.shape

        f = []
        for t in range(1, seq_dim):
            x_t = x[:, t, 1:height - 1, 1:width - 1, self.input_dim]
            f_a = x_t - x[:, t-1, :height-2, 1:width-1, self.input_dim]
            f_b = x_t - x[:, t-1, 2:height, 1:width-1, self.input_dim]
            f_c = x_t - x[:, t-1, 1:height-1, :width-2, self.input_dim]
            f_d = x_t - x[:, t-1, 1:height-1, 2:width, self.input_dim]
            f_t = torch.stack([f_a + f_b, f_c + f_d], dim=-1)
            f.append(f_t)
        f = torch.stack(f, dim=1)

        f_x = f[:, :self.window_in_len]

        return f_x

    @staticmethod
    def __load_batch(batch):
        """
        loads from the path and creates the batch.

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
