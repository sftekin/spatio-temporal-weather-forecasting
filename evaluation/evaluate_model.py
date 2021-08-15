import os
import glob
import shutil
import pickle as pkl

import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, \
    mean_absolute_percentage_error, r2_score

from config import data_params
from data_creator import DataCreator
from batch_generator import BatchGenerator


def calc_eval_metrics(pred, y):
    sequance_len = pred.shape[1]
    metrics = [mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score]
    metric_scores = np.zeros((sequance_len, len(metrics)))
    for i in range(sequance_len):
        pred_step, y_step = pred[:, i], y[:, i]
        for j, metric in enumerate(metrics):
            metric_scores[i, j] = metric(y_true=y_step.flatten(), y_pred=pred_step.flatten())

    return metric_scores


def run_model(model, batch_generator, mode, device):
    model.to(device)
    batch_size = batch_generator.dataset_params['batch_size']
    sequence_len = batch_generator.dataset_params["window_out_len"]
    metric_arr = np.zeros((sequence_len, 4))  # Since we have 4 metrics for each time step
    running_mse = 0
    criterion = nn.MSELoss()
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
    print('{} finished, MSE: {:.5f}'.format(mode, running_mse))

    return metric_arr


def check_point(model_name, scores, save_dir):
    file_name = f"{model_name}_evaluation_metrics.pkl"
    file_path = os.path.join(save_dir, file_name)
    with open(file_path, "wb") as f:
        pkl.dump(scores, f)


def get_scores(checkpoint_dir):
    figures_dir = "figures"
    eval_figures_dir = os.path.join(figures_dir, "evaluation_figures")
    if not os.path.exists(eval_figures_dir):
        os.makedirs(eval_figures_dir)

    model_metrics = {}
    val_exp_df = pd.DataFrame()
    test_exp_df = pd.DataFrame()
    for score_path in glob.glob(os.path.join(checkpoint_dir, "*.pkl")):
        model_name = os.path.basename(score_path).split("evaluation_metrics")[0].rstrip("_")
        with open(score_path, "rb") as f:
            metrics = pkl.load(f)
        model_metrics[model_name] = metrics

        val_exp_scores = get_experiment_scores(metrics, model_name, score_type="val")
        test_exp_scores = get_experiment_scores(metrics, model_name, score_type="test")
        val_exp_df = pd.concat([val_exp_df, val_exp_scores], axis=1)
        test_exp_df = pd.concat([test_exp_df, test_exp_scores], axis=1)

    val_exp_df.to_csv(os.path.join(evaluation_dir, "val_exp_scores.csv"))
    test_exp_df.to_csv(os.path.join(evaluation_dir, "test_exp_scores.csv"))

    plot_scores(model_metrics, score_type="val_score", save_dir=eval_figures_dir)
    plot_scores(model_metrics, score_type="test_score", save_dir=eval_figures_dir)

    print()


def get_experiment_scores(metrics, model_name, score_type):
    data_length = 24
    start_date_str = "2000-01-01"
    val_ratio = 0.1
    test_ratio = 0.1
    stride = 6
    range_index = []
    start_date = pd.to_datetime(start_date_str)
    for i in range(10):
        end_date = start_date + pd.DateOffset(months=data_length) - pd.DateOffset(hours=1)
        r = pd.date_range(start_date, end_date, freq="3H")
        data_len = len(r)
        val_count = int(data_len * val_ratio)
        test_count = int(data_len * test_ratio)

        train_count = data_len - val_count - test_count
        val_date_range = r[train_count:train_count+val_count]
        test_date_range = r[train_count+val_count:]

        val_date_range_str = val_date_range[0].strftime("%Y-%m") + "_" + \
                             val_date_range[-1].strftime("%Y-%m")
        test_date_range_str = test_date_range[0].strftime("%Y-%m") + "_" + \
                              test_date_range[-1].strftime("%Y-%m")

        if score_type == "test":
            range_index.append(test_date_range_str)
        else:
            range_index.append(val_date_range_str)

        start_date += pd.DateOffset(months=stride)

    metrics_order = ["MAE", "MSE", "MAPE", "R2"]
    metrics_order = [m + "_" + model_name for m in metrics_order]
    exp_scores = metrics[f"{score_type}_score"]
    experiment_values = []
    for exp_id in range(10):
        metric_arr = exp_scores[exp_id]
        experiment_values.append(np.mean(metric_arr, axis=0))
    experiment_values = np.stack(experiment_values, axis=0)
    experiment_df = pd.DataFrame(experiment_values, columns=metrics_order, index=range_index)

    return experiment_df


def plot_scores(model_metrics, score_type, save_dir):
    fig, ax = plt.subplots(figsize=(5, 4))
    colors = ["r", "b", "g"]
    marker = ["D", "o", "X"]
    for i, model_name in enumerate(["weather_model", "convlstm", "u_net"]):
        scores = model_metrics[model_name]
        metrics = scores[score_type]

        mean_mse = np.mean(metrics, axis=0)[:, 0]
        std_mse = np.std(metrics, axis=0)[:, 0]

        t_value = 2.262
        standard_err = std_mse / np.sqrt(10)
        conf_err = standard_err * t_value

        ax.errorbar(range(1, len(mean_mse)+1), mean_mse,
                    yerr=conf_err, fmt=marker[i], color=colors[i],
                    capsize=5, capthick=3, markersize=5, label=model_name)
    y_ticks = np.linspace(0, 2, 11)
    y_labels = ["{:.2f}".format(i) for i in y_ticks]
    ax.set_xticks(range(1, 11))
    ax.set_yticks(y_ticks)
    ax.set_ylabel(y_labels)
    ax.set_xlabel("Number of steps forward")
    ax.set_ylabel("MAE")
    ax.set_title(f"{score_type.replace('_', ' ')} comparision")
    ax.grid(True)
    ax.legend()
    plt.savefig(os.path.join(save_dir, f"{score_type}.png"), dpi=200)


def evaluate(rebuild_data, cp_dir):
    results_dir = "results"
    model_name = "weather_model"

    dump_file_dir = os.path.join('data', 'data_dump')
    exp_dir_paths = os.path.join(results_dir, model_name, "exp*")
    exp_dirs = list(glob.glob(exp_dir_paths))
    exp_nums = [int(os.path.basename(d).split("_")[-1]) for d in exp_dirs]
    exp_dirs = np.array(exp_dirs)[np.argsort(exp_nums)]

    count = 0
    val_scores, test_scores = [], []
    for file in exp_dirs:
        if count >= 10:
            break
        model_path = os.path.join(file, "model.pkl")
        loss_path = os.path.join(file, "loss.pkl")
        config_path = os.path.join(file, "config.pkl")

        # load loss
        with open(loss_path, "rb") as f:
            loss = pkl.load(f)

        # load model
        with open(model_path, "rb") as f:
            model = pkl.load(f)

        try:
            # load config
            with open(config_path, "rb") as f:
                config = pkl.load(f)

            # parse config
            experiment_params = config["experiment_params"]
            model_params = config[f"{model_name}_params"]
            data_length = experiment_params["data_length"]

        except FileNotFoundError:
            from config import experiment_params, model_params
            data_length = experiment_params["data_length"]
            model_params = model_params[model_name]

        # create data
        try:
            start_date_str, end_date_str = loss[-1].split("_")
            start_date = pd.to_datetime(start_date_str)

        except Exception:
            start_year, start_month, last_str = loss[-1].split("-")[:3]
            start_day = last_str[:2]
            start_date = pd.to_datetime(f"{start_year}-{start_month}-{start_day}")

        end_date = start_date + pd.DateOffset(months=data_length) - pd.DateOffset(hours=1)
        date_range_str = start_date.strftime("%Y-%m-%d") + "_" + end_date.strftime("%Y-%m-%d")

        data_params["rebuild"] = rebuild_data
        data_creator = DataCreator(start_date=start_date, end_date=end_date, **data_params)
        weather_data = data_creator.create_data()

        print(f"Evaluating experiment-{file} where the date range is {date_range_str}")

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

        mode = "val"
        val_metrics = run_model(model, batch_generator, mode, device)
        val_scores.append(val_metrics)

        mode = "test"
        test_metrics = run_model(model, batch_generator, mode, device)
        test_scores.append(test_metrics)

        print(f"Evaluation of {file} finished")
        scores = {"val_score": val_scores, "test_score": test_scores}
        check_point(model_name, scores, cp_dir)

        # remove dump directory
        shutil.rmtree(dump_file_dir)
        count += 1


if __name__ == '__main__':
    evaluation_dir = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
    checkpoints_dir = os.path.join(evaluation_dir, "checkpoints")
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)

    # evaluate(rebuild_data=True, cp_dir=checkpoints_dir)

    get_scores(checkpoints_dir)

