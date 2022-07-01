import os
import pickle as pkl

import pandas as pd
import numpy as np

from experimenter import get_experiment_elements, log_results
from data_creator import DataCreator
from batch_generator import BatchGenerator


# TODO: Make here a class, write a predict loop with forecast horizon and iterative mode. Ask lat array if weighted true

def inference_on_test(model_name, device, exp_num, test_data_folder, start_date_str, end_date_str, forecast_horizon):
    trainer, model, dumped_generator = get_experiment_elements(model_name, device, exp_num)

    start_date = pd.to_datetime(start_date_str)
    end_date = pd.to_datetime(end_date_str) - pd.DateOffset(hours=1)

    path_list = DataCreator.get_file_paths(test_data_folder)
    path_arr = DataCreator.sort_files_by_date(paths=path_list,
                                              start_date=start_date,
                                              end_date=end_date)

    normalize_flag = dumped_generator.normalize_flag
    params = dumped_generator.dataset_params
    params["stride"] = params["window_out_len"]
    batch_generator = BatchGenerator(weather_data=path_arr, val_ratio=0.0, test_ratio=1.0,
                                     normalize_flag=normalize_flag, params=params)

    print("-*-" * 20)
    print(f"Inference on {test_data_folder} between {start_date_str} and {end_date_str} dates")
    test_loss, test_metric = trainer.predict(model, batch_generator)

    ts_metrics, all_metrics = calc_weighted_metrics(model, batch_generator, device)
    test_metric["WeightedMAE"] = all_metrics[0]
    test_metric["WeightedRMSE"] = all_metrics[1]
    test_metric["WeightedACC"] = all_metrics[2]

    # log the results
    log_results(scores={"inference-test": test_metric},
                trainer=trainer,
                date_range_str=f"{start_date_str}_{end_date_str}")

    # save metrics
    metric_dir = os.path.join('results', model_name, f"exp_{exp_num}", "test_inference_scores.pkl")
    with open(metric_dir, "wb") as f:
        pkl.dump(test_metric, f)


def calc_weighted_metrics(model, generator, device):
    model.to(device)
    model.eval()
    running_preds, running_labels, weights = [], [], None
    for idx, (x, y) in enumerate(generator.generate("test")):
        if hasattr(model, 'hidden'):
            hidden = model.init_hidden(batch_size=x.shape[0])
        else:
            hidden = None
        # (b, t, m, n, d) -> (b, t, d, m, n)
        x = x.permute(0, 1, 4, 2, 3).float().to(device)
        y = y.permute(0, 1, 4, 2, 3).float().to(device)

        pred = model.forward(x=x.clone(), hidden=hidden)

        if generator.normalizer:
            pred = generator.normalizer.inv_norm(pred, device)
            y = generator.normalizer.inv_norm(y, device)

        # get latitude array
        if idx == 0:
            min_lat, max_lat = generator.normalizer.min_max[17]
            lats_arr = x[0, 0, -2].detach().cpu().numpy() * max_lat[0, 0].numpy() + min_lat[0, 0].numpy()
            weights = np.cos(np.deg2rad(lats_arr))
            weights /= weights[:, 0].mean()
            weights = np.expand_dims(weights, axis=0)

        # store the pred and target
        running_preds.append(pred.detach().cpu().numpy())
        running_labels.append(y.detach().cpu().numpy())

    ts_metrics, all_metrics = _calc_weighted_diff_meterics(running_preds, running_labels, weights)

    return ts_metrics, all_metrics


def _calc_weighted_diff_meterics(pred, target, weights):
    pred = np.concatenate(pred, axis=0)
    target = np.concatenate(target, axis=0)
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
