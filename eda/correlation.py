import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from eda.plot_helper import load_dump_data


save_path = 'figures'
all_features = ['divergence', 'fraction of cloud cover', 'geopotential', 'potential vorticity', 'relative humidity',
                'specific cloud ice water content', 'specific cloud liquid water content', 'Specific humidity',
                'specific rain water content', 'specific snow water content', 'temperature', 'u-wind',
                'v-wind', 'vertical velocity']
feat_idx = [0, 2, 3, 4, 7, 10, 11, 12, 13]
selected_features = [all_features[i] for i in feat_idx]


data_arr, date_list = load_dump_data()
_, m_dim, n_dim, _ = data_arr.shape

all_corrs = []
for m in range(m_dim):
    for n in range(n_dim):

        selected_cell = data_arr[:, m, n, feat_idx]
        cell_df = pd.DataFrame(selected_cell, columns=[selected_features])
        corr = cell_df.corr()

        all_corrs.append(corr.values)

corr = np.mean(all_corrs, axis=0)

# corr_df = pd.DataFrame(data=corr, index=selected_features, columns=selected_features)

fig = plt.figure(figsize=(7, 7))
plt.imshow(corr)
ticks = range(len(selected_features))
plt.xticks(ticks, selected_features, rotation=45, size=10)
plt.yticks(ticks, selected_features, size=10)

x, y = corr.shape
for m in range(x):
    for n in range(y):
        x_loc = ticks[m] - 1/4
        y_loc = ticks[n]
        score_txt = "{:.2f}".format(corr[m, n])
        plt.text(x_loc, y_loc, score_txt, size=9)

fig.savefig(f"{save_path}/correlation.png", bbox_inches="tight", pad_inches=0, dpi=300)




