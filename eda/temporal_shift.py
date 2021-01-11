import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from eda.plot_helper import load_dump_data

start_month = 8
day_count = 60
start_idx = start_month * 30 * 6
end_idx = start_idx + 6 * day_count
day_shift = 8
week_shift = 56
month_shift = 240
fontsize = 12
save_path = "figures"

data_arr, date_list = load_dump_data(start_idx=start_idx, end_idx=end_idx)
temperature = data_arr[:, 30, 60, 10]

fig = plt.figure(figsize=(10, 5))
for shift in [day_shift, week_shift, month_shift]:
    abs_arr = temperature[shift:]
    prev_arr = temperature[:-shift]
    x_axis = np.arange(len(abs_arr))
    x_ticks = [i for i in range(0, len(abs_arr)+day_shift, day_shift)]
    x_labels = [f"day_{i//day_shift}" for i in x_ticks]
    plt.plot(x_axis, abs_arr, '-2', color='r', label='current_value')
    plt.plot(x_axis, prev_arr, '-2', color='b', label='delayed_value')

    plt.xlabel("Days", fontsize=fontsize)
    plt.ylabel("Temperature (K)", fontsize=fontsize)
    plt.title(f"Timeshift for {shift // 8} Day", fontsize=fontsize)
    plt.xlim(min(x_ticks), max(x_axis))
    plt.xticks(x_ticks, x_labels, rotation=30, size=6, horizontalalignment='center', verticalalignment='top')
    plt.legend(fontsize=fontsize)
    plt.grid(True)
    fig.savefig(f"{save_path}/temperature_{shift}.png", bbox_inches="tight", pad_inches=0, dpi=300)
    fig.clf()

