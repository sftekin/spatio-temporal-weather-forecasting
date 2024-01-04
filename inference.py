import os
import pickle as pkl

import pandas as pd
import numpy as np
import torch
from torchmetrics.functional import mean_absolute_percentage_error, \
    mean_absolute_error, mean_squared_error

from experimenter import get_experiment_elements, log_results
from data_generation.data_creator import DataCreator
from data_generation.batch_generator import BatchGenerator


def inference_on_test(model_name, device, exp_num, test_data_folder, start_date_str, end_date_str, forecast_horizon,
                      selected_dim, exp_dir, dataset_type, exp_type):
    trainer, model, dumped_generator = get_experiment_elements(model_name, device, exp_num, exp_dir)

    start_date = pd.to_datetime(start_date_str)
    end_date = pd.to_datetime(end_date_str) - pd.DateOffset(hours=1)

    path_list = DataCreator.get_file_paths(test_data_folder)
    path_arr = DataCreator.sort_files_by_date(paths=path_list,
                                              start_date=start_date,
                                              end_date=end_date)

    normalize_flag = dumped_generator.normalize_flag
    window_out_len = dumped_generator.dataset_params["window_out_len"]
    params = dumped_generator.dataset_params
    params["stride"] = forecast_horizon
    if exp_type != "direct":
        params["window_out_len"] = forecast_horizon
    batch_generator = BatchGenerator(weather_data=path_arr, val_ratio=0.0, test_ratio=1.0,
                                     normalize_flag=normalize_flag, params=params)

    print("-*-" * 20)
    print(f"Inference on {test_data_folder} between {start_date_str} and {end_date_str} dates")
    if window_out_len < forecast_horizon:
        print(f"Performing iterative prediction since window_out "
              f"({window_out_len}) < forecast_horizon ({forecast_horizon})")

    with torch.no_grad():
        ts_metrics, all_metrics = calc_metric_scores(model, batch_generator, device, selected_dim, dataset_type)

    # log the results
    log_results(scores={"inference-test": all_metrics},
                trainer=trainer,
                date_range_str=f"{start_date_str}_{end_date_str}")

    # save metrics
    metric_dir = os.path.join(exp_dir, model_name, f"exp_{exp_num}", "test_scores.pkl")
    scores = {"ts_metrics": ts_metrics, "all_metrics": all_metrics}
    with open(metric_dir, "wb") as f:
        pkl.dump(scores, f)


def calc_metric_scores(model, generator, device, selected_dim, dataset_type):
    model.to(device)
    model.eval()
    running_preds, running_labels, weights = [], [], None
    for idx, (x, y) in enumerate(generator.generate("test")):
        print('\r\t{}:{}/{}'.format("test", idx, generator.num_iter("test")),
              flush=True, end='')
        if hasattr(model, 'hidden'):
            hidden = model.init_hidden(batch_size=x.shape[0])
        else:
            hidden = None
        # (b, t, m, n, d) -> (b, t, d, m, n)
        x = x.permute(0, 1, 4, 2, 3).float().to(device)
        y = y.permute(0, 1, 4, 2, 3).float().to(device)

        # get prediction
        pred = model.forward(x=x.clone(), hidden=hidden)
        if pred.shape[1] < y.shape[1]:
            # iterative mode
            pred_list = [pred.clone()]
            for _ in range(pred.shape[1], y.shape[1], pred.shape[1]):
                x = pred
                hidden = model.init_hidden(batch_size=x.shape[0]) if hidden is not None else None
                pred = model.forward(x=x.clone(), hidden=hidden)
                pred_list.append(pred)
            pred = torch.cat(pred_list, dim=1)

        if generator.normalizer:
            pred = generator.normalizer.inv_norm(pred, device)
            y = generator.normalizer.inv_norm(y, device)

        # store the pred and target
        running_preds.append(pred[:, :, [selected_dim]].detach().cpu())
        running_labels.append(y[:, :, [selected_dim]].detach().cpu())

    # concat all pred and targets
    pred_all = torch.cat(running_preds, dim=0)
    target_all = torch.cat(running_labels, dim=0)

    ts_metrics, all_metrics = _calc_metrics(pred=pred_all, target=target_all)
    if dataset_type == "weatherbench":
        # calculate weights for weighted metrics
        lats_arr = np.load(os.path.join("resources", "lat_arr.npy"))
        weights = np.cos(np.deg2rad(lats_arr))
        weights /= weights[:, 0].mean()
        weights = np.expand_dims(weights, axis=0)

        ts_weighted_scores, all_weighted_scores = _calc_weighted_meterics(pred_all.numpy(),
                                                                          target_all.numpy(), weights)

        weighted_metric_names = ["WeightedMAE", "WeightedRMSE", "WeightedACC"]
        for i, m_name in enumerate(weighted_metric_names):
            all_metrics[m_name] = all_weighted_scores[i]
            ts_metrics[m_name] = ts_weighted_scores[:, i]

    return ts_metrics, all_metrics


def _calc_weighted_meterics(pred, target, weights):
    batch_count, seq_len, d_dim, height, width = pred.shape

    # calculate mae, rmse
    error = pred - target
    error_flatten = error.reshape((-1, height, width))
    weighted_ae = np.abs(error_flatten) * weights
    weighted_se = (error_flatten ** 2) * weights

    # calculate acc
    pred_flat = pred.reshape((-1, height, width))
    target_flat = target.reshape((-1, height, width))
    climatology = np.mean(target_flat, axis=0, keepdims=True)
    pred_diff = pred_flat - climatology
    target_diff = target_flat - climatology
    all_acc = (pred_diff * target_diff * weights).sum() / \
              np.sqrt((weights * pred_diff ** 2).sum() * (weights * target_diff ** 2).sum())

    all_mae, all_rmse = 0, 0
    for i in range(error_flatten.shape[0]):
        all_rmse += np.sqrt(weighted_se[i].mean())
        all_mae += weighted_ae[i].mean()
    all_mae /= (i + 1)
    all_rmse /= (i + 1)

    weighted_ae = weighted_ae.reshape(batch_count, seq_len, height, width)
    weighted_se = weighted_se.reshape(batch_count, seq_len, height, width)
    pred_diff = pred_diff.reshape(batch_count, seq_len, height, width)
    target_diff = target_diff.reshape(batch_count, seq_len, height, width)
    ts_metrics = np.zeros((seq_len, 3))
    for t in range(seq_len):
        ts_metrics[t, 0] += weighted_ae[:, t].mean()
        ts_metrics[t, 1] += np.sqrt(weighted_se[:, t].mean())
        ts_metrics[t, 2] += (pred_diff[:, t] * target_diff[:, t] * weights).sum() / \
                            np.sqrt((weights * pred_diff[:, t] ** 2).sum() * (weights * target_diff[:, t] ** 2).sum())

    return ts_metrics, (all_mae, all_rmse, all_acc)


def _calc_metrics(pred, target):
    metric_collection = {
        "MSE": mean_squared_error,
        "MAE": mean_absolute_error,
        "MAPE": mean_absolute_percentage_error,
        "RMSE": lambda preds, target: torch.sqrt(mean_squared_error(preds, target))
    }

    all_metrics, ts_metrics = {}, {}
    for key, func in metric_collection.items():
        all_metrics[key] = func(preds=pred, target=target).numpy()
        ts_list = []
        for t in range(pred.shape[1]):
            ts_list.append(func(preds=pred[:, t].contiguous(), target=target[:, t].contiguous()).numpy())
        ts_metrics[key] = ts_list

    return ts_metrics, all_metrics
