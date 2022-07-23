import numpy as np
import matplotlib.pyplot as plt

from eda.plot_helper import load_dump_data, create_flow_mat

save_path = "figures"
arrw_strd = 3
data_arr, date_list = load_dump_data()

f = create_flow_mat(data_arr[:4], input_dim=10)

seq_len, m, n, _ = f.shape
X, Y = np.meshgrid(np.arange(0, n), np.arange(0, m))

fig, axs = plt.subplots(1, seq_len, figsize=(10, 3))
for t in range(seq_len):
    U = f[t, :, :, 0]
    V = f[t, :, :, 1]

    q = axs[t].quiver(X[::arrw_strd, ::arrw_strd],
                      Y[::arrw_strd, ::arrw_strd],
                      U[::arrw_strd, ::arrw_strd],
                      V[::arrw_strd, ::arrw_strd])
    axs[t].set_xlim(0, n)
    axs[t].set_ylim(0, m)


fig.savefig(f"{save_path}/flow_stride_{arrw_strd}.png", bbox_inches="tight", pad_inches=0, dpi=300)


