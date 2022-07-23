import os
import pandas as pd
import numpy as np


def load_dump_data(start_idx=0, end_idx=-1):
    read_path = "data/data_dump"

    date_list = []
    for file in os.listdir(read_path):
        date_list.append(pd.to_datetime(file.split('.')[0], format="%Y-%m-%d_%H"))
    date_list = sorted(date_list)[start_idx:end_idx]

    file_list = [date.strftime("%Y-%m-%d_%H") + ".npy" for date in date_list]

    data_arr = []
    for i, file in enumerate(file_list):
        file_path = os.path.join("data/data_dump", file)
        data_arr.append(np.load(file_path))
    data_arr = np.stack(data_arr, axis=0)

    return data_arr, date_list


def create_flow_mat(x, input_dim):
    """

    :param x: (T+1, M, N, D)
    :return:
    :rtype:
    """
    seq_dim, height, width, d_dim = x.shape

    f = []
    for t in range(1, seq_dim):
        x_t = x[t, 1:height - 1, 1:width - 1, input_dim]
        f_a = x_t - x[t-1, :height-2, 1:width-1, input_dim]
        f_b = x_t - x[t-1, 2:height, 1:width-1, input_dim]
        f_c = x_t - x[t-1, 1:height-1, :width-2, input_dim]
        f_d = x_t - x[t-1, 1:height-1, 2:width, input_dim]
        f_t = np.stack([f_a + f_b, f_c + f_d], axis=-1)
        f.append(f_t)
    f = np.stack(f, axis=0)

    return f

