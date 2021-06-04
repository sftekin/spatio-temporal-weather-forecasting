import os
import glob
import pickle as pkl
import numpy as np

results_dir = "../results"
model_name = "weather_model"
exp_dir_paths = os.path.join(results_dir, model_name, "exp*")
exp_dirs = list(glob.glob(exp_dir_paths))
exp_nums = [int(os.path.basename(d).split("_")[-1]) for d in exp_dirs]
exp_dirs = np.array(exp_dirs)[np.argsort(exp_nums)]

for file in exp_dirs:
    loss_path = file + "/loss.pkl"
    with open(loss_path, "rb") as f:
        loss = pkl.load(f)
    print(file, loss[4], loss[2], loss[3])

