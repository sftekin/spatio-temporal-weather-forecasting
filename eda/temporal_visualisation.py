import numpy as np
import matplotlib.pyplot as plt

from eda.plot_helper import load_dump_data

data_arr, date_list = load_dump_data()
temperature = data_arr[:, 30, 60, 10]
save_path = "figures"


x_axis = np.arange(len(temperature))
x_ticks = [i for i in range(0, len(temperature), 240)]
x_labels = [date_list[i].strftime("%Y-%m") for i in x_ticks]

fig = plt.figure(figsize=(10, 5))
plt.plot(x_axis, temperature, color='r')
plt.ylabel("Temperature (K)", fontsize=8)
plt.xticks(x_ticks, x_labels, rotation=30, size=8, horizontalalignment='center', verticalalignment='top')
plt.title(f"Seasonal Data from {min(date_list).year} to {max(date_list).year}")
plt.grid(True)
fig.savefig(f"{save_path}/temperature_seasonal.png", bbox_inches="tight", pad_inches=0, dpi=300)

