import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf

from eda.plot_helper import load_dump_data


dpi = 300
start_month = 8
day_count = 60
start_idx = start_month * 30 * 6
end_idx = start_idx + 6 * day_count

data_arr, date_list = load_dump_data(start_idx=start_idx, end_idx=end_idx)
temperature = data_arr[:, 20, 30, 10]

fig, ax = plt.subplots(figsize=(10, 8))
plot_acf(temperature, ax=ax, lags=50)
plt.savefig("figures/acf.png", dpi=dpi)

fig, ax = plt.subplots(figsize=(10, 8))
plot_pacf(temperature, ax=ax, lags=50)
ax.set_xlabel("Lags")
ax.set_ylabel("Partial Correlation")
ax.set_title("PACF")
plt.savefig("figures/pacf.png", dpi=dpi)


