import os
import glob
import pickle as pkl

import numpy as np
import pandas as pd
import torch.nn as nn

from data_creator import DataCreator
from batch_generator import BatchGenerator



results_dir = "../results"
model_name = "weather_model"
exp_dir_paths = os.path.join(results_dir, model_name, "exp*")
exp_dirs = list(glob.glob(exp_dir_paths))
exp_nums = [int(os.path.basename(d).split("_")[-1]) for d in exp_dirs]
exp_dirs = np.array(exp_dirs)[np.argsort(exp_nums)]

for file in exp_dirs:
    model_path = os.path.join(file, "model.pkl")
    loss_path = os.path.join(file, "loss.pkl")
    config_path = os.path.join(file, "config.pkl")

    # load loss
    with open(loss_path, "rb") as f:
        loss = pkl.load(f)

    # load config
    with open(config_path, "rb") as f:
        config = pkl.load(f)

    # load model
    with open(model_path, "rb") as f:
        model = pkl.load(f)

    # parse config
    data_params = config["data_params"]
    experiment_params = config["experiment_params"]
    model_params = config[f"{model_name}_params"]
    data_params["target_dim"] = 10
    data_params["downsample_mode"] = "selective"

    # create data
    start_date_str, end_date_str = loss[-1].split("_")
    start_date = pd.to_datetime(start_date_str)
    end_date = pd.to_datetime(end_date_str)

    data_creator = DataCreator(start_date=start_date, end_date=end_date, **data_params)
    weather_data = data_creator.create_data()

    # create batch gen
    val_ratio = experiment_params['val_ratio']
    test_ratio = experiment_params['test_ratio']
    normalize_flag = experiment_params['normalize_flag']
    device = experiment_params['device']
    batch_gen_params = model_params["batch_gen"]

    batch_generator = BatchGenerator(weather_data=weather_data,
                                     val_ratio=val_ratio,
                                     test_ratio=test_ratio,
                                     params=batch_gen_params,
                                     normalize_flag=normalize_flag)

    criterion = nn.MSELoss()
    batch_size = batch_generator.dataset_params['batch_size']
    for idx, (x, y, f_x) in enumerate(batch_generator.generate('val')):
        print('\r{}:{}/{}'.format('val', idx, batch_generator.num_iter('val')),
              flush=True, end='')

        if hasattr(model, 'hidden'):
            hidden = model.init_hidden(batch_size)
        else:
            hidden = None

        x = x.float().to(device)
        y = y.float().to(device)

        # (b, t, m, n, d) -> (b, t, d, m, n)
        x = x.permute(0, 1, 4, 2, 3)
        y = y.permute(0, 1, 4, 2, 3)

        pred = model.forward(x=x, f_x=f_x, hidden=hidden)

        if batch_generator.normalizer:
            pred = batch_generator.normalizer.inv_norm(pred, device)
            y = batch_generator.normalizer.inv_norm(y, device)

        loss = criterion(pred, y)
        print()

