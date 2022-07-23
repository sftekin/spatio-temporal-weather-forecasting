"""
This script plots the metrics of weatherbench experiments
"""

import os
import pickle as pkl

import numpy as np
import matplotlib.pyplot as plt

from experimenter import get_exp_count


def get_model_scores(in_dir):
    model2score = {}
    for model_name in os.listdir(in_dir):
        exp_count = get_exp_count(model_name, result_dir=in_dir)
        if exp_count > 1:
            print(f"More than one experiment results found in"
                  f" {os.path.join(in_dir, model_name)} using the last one")
        last_exp_dir = os.path.join(in_dir, model_name, f"exp_{exp_count}")
        with open(os.path.join(last_exp_dir, "test_scores.pkl"), "rb") as f:
            scores = pkl.load(f)
        model2score[model_name] = scores

    return model2score


def plot_metric(metric_name, score_dict, model_colors):
    fig, ax = plt.subplots(figsize=(8, 5))
    color_count = len(score_dict.keys()) - 1
    for model_name, scores in score_dict.items():
        ts_score = scores["ts_metrics"][metric_name]
        x_axis = np.arange(len(ts_score))
        ax.plot(x_axis, ts_score, label=model_name, lw=3, color=model_colors[model_name])
        ax.set_xlabel("Forecast time (hours)")
        ax.set_ylabel(f"T850 {metric_name.replace('Weighted', '')} (K)")
        color_count -= 1
    return fig, ax


def plot_direct_scores(ax, metric_name, scores, model_colors):
    x_axis = np.arange(1, len(scores[metric_name]) + 1) * 24
    ax.scatter(x_axis, scores[metric_name], label="weather_model (direct)", lw=2,
               color=model_colors["weather_model (iterative)"], s=80, edgecolors="k")
    ax.legend(framealpha=0.5, frameon=True, edgecolor='k')


def plot_exps():
    iter_res_path = os.path.join("results", "iterative_results")
    seq_res_path = os.path.join("results", "sequential_results")
    direct_res_path = os.path.join("results", "direct_results")

    seq_scores = get_model_scores(seq_res_path)
    iter_scores = get_model_scores(iter_res_path)
    direct_scores = get_model_scores(direct_res_path)

    plt.style.use("seaborn")

    best_model_scores = {}
    for model_name in seq_scores.keys():
        seq_rmse = seq_scores[model_name]['all_metrics']["WeightedRMSE"]
        iter_rmse = iter_scores[model_name]['all_metrics']["WeightedRMSE"]
        if seq_rmse < iter_rmse:
            best_model_scores[f"{model_name} (sequential)"] = seq_scores[model_name]
        else:
            best_model_scores[f"{model_name} (iterative)"] = iter_scores[model_name]

    model_colors = {model_name: c for model_name, c in
                    zip(best_model_scores.keys(), ["tab:blue", "tab:orange", "tab:green", "tab:red"])}

    for metric_name in ["WeightedRMSE", "WeightedMAE", "WeightedACC"]:
        fig, ax = plot_metric(metric_name, best_model_scores, model_colors)
        plot_direct_scores(ax, metric_name, direct_scores["weather_model"]["ts_metrics"], model_colors)
        plt.savefig(os.path.join("figures", f"{metric_name}.png"), bbox_inches="tight", dpi=300)

    print()


if __name__ == '__main__':
    plot_exps()
