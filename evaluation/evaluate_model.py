import os
import glob
import pickle as pkl

import numpy as np
import pandas as pd
import torch.nn as nn

from data_creator import DataCreator
from batch_generator import BatchGenerator
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score


def calc_eval_metrics(pred, y):
    sequance_len = pred.shape[1]
    metrics = [mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score]
    metric_scores = np.zeros((sequance_len, len(metrics)))
    for i in range(sequance_len):
        pred_step, y_step = pred[:, i], y[:, i]
        for j, metric in enumerate(metrics):
            metric_scores[i, j] = metric(y_true=y_step.flatten(), y_pred=pred_step.flatten())

    return metric_scores


def evaluate(rebuild_data=False):
    results_dir = "results"
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
        data_params["rebuild"] = rebuild_data
        data_length = experiment_params["data_length"]

        # create data
        start_date_str, end_date_str = loss[-1].split("_")
        start_date = pd.to_datetime(start_date_str)
        end_date = start_date + pd.DateOffset(months=data_length) - pd.DateOffset(hours=1)

        data_creator = DataCreator(start_date=start_date, end_date=end_date, **data_params)
        weather_data = data_creator.create_data()

        # create batch gen
        val_ratio = experiment_params['val_ratio']
        test_ratio = experiment_params['test_ratio']
        normalize_flag = experiment_params['normalize_flag']
        device = experiment_params['device']
        batch_gen_params = model_params["batch_gen"]
        batch_gen_params["shuffle"] = False

        batch_generator = BatchGenerator(weather_data=weather_data,
                                         val_ratio=val_ratio,
                                         test_ratio=test_ratio,
                                         params=batch_gen_params,
                                         normalize_flag=normalize_flag)

        model.to(device)
        batch_size = batch_generator.dataset_params['batch_size']
        sequence_len = batch_generator.dataset_params["window_out_len"]
        metric_arr = np.zeros((sequence_len, 4))  # Since we have 4 metrics for each time step
        running_mse = 0
        criterion = nn.MSELoss()
        mode = "val"
        for idx, (x, y, f_x) in enumerate(batch_generator.generate(mode)):
            print('\r{}:{}/{}'.format(mode, idx, batch_generator.num_iter(mode)),
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

            running_mse += criterion(pred, y).detach().cpu().numpy()

            pred, y = pred.detach().cpu().numpy(), y.detach().cpu().numpy()
            metric_step = calc_eval_metrics(pred, y)
            metric_arr += metric_step

        metric_arr /= (idx + 1)
        running_mse /= (idx + 1)
        print(running_mse)


if __name__ == '__main__':
    evaluate()
