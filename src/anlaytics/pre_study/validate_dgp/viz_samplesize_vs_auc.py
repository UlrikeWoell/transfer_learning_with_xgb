import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numpy import random as npr


def get_json(path: str):
    with open(path) as fp:
        data = json.load(fp)
    return data


def get_tuning_result(file_path: str):
    return get_json(f"{file_path}/tuning_results.json")


def get_train_log(file_path: str):
    return get_json(f"{file_path}/train_log.json")


def normalized_gain(initial_val: float, final_val: float):
    return (final_val - initial_val) / (1 - initial_val)


def sample_size_auc():
    directory = "data/validate_dgp"
    folders = sorted(os.listdir(directory))
    data = []
    fails = []

    for folder in folders:
        try:
            tuning = get_tuning_result(f"{directory}/{folder}")
            log = get_train_log(f"{directory}/{folder}")
            data.append(
                {
                    "Sample size": log["sample"]["n"] * 1.0,
                    "experiment": log["domain"]["correllation_matrix"]["seed"],
                    "ROC-AUC": tuning["auc_roc"],
                }
            )
        except Exception:
            fails.append(folder)

    return pd.DataFrame(data)


def add_red_points_to_data(df: pd.DataFrame):
    red_points = [200, 2000]
    space = -0.025
    df["dot_label"] = np.where(
        df["Sample size"].isin(red_points), round(df["ROC-AUC"], 3), np.NaN
    )
    df["dot_position"] = np.where(
        df["Sample size"].isin(red_points), df["ROC-AUC"] + space, 0
    )
    return df


def make_plot():
    df = sample_size_auc()
    df = add_red_points_to_data(df)
    experiments = list(set(df["experiment"]))[0:16]

    sns.set_theme()

    fig, axes = plt.subplots(4, 4, figsize=(8, 8), sharex=True, sharey=True)
    fig.supxlabel('Sample size', fontsize = 9)
    fig.supylabel('ROC-AUC', fontsize = 9)
    for i, experiment in enumerate(experiments):
        ax = axes.flat[i]
        subset = df[df["experiment"] == experiment]

        auc200 = subset[subset["Sample size"] == 200]["ROC-AUC"].iloc[0]
        auc2000 = subset[subset["Sample size"] == 2000]["ROC-AUC"].iloc[0]
        gain = normalized_gain(auc200, auc2000)

        ax.scatter(data=subset, x="Sample size", y="ROC-AUC", s=10)

        for _, row in subset.iterrows():
            if (row["Sample size"] == 200) | (row["Sample size"] == 2000):
                ax.scatter(row["Sample size"], row["ROC-AUC"], color="brown")
                ax.text(
                    row["Sample size"],
                    row["dot_position"],
                    f"{row['ROC-AUC']:.3f}",
                    color="black",
                    verticalalignment="bottom",
                    horizontalalignment="left",
                    fontsize=9,
                )
        ax.text(
            x=2520,
            y=0.825,
            s=f"NG={gain:.2f}",
            color="black",
            verticalalignment="center",
            horizontalalignment="center",
            fontsize=9,
        )            
        ax.set_xticks([200,2000], minor=False)
        ax.set_xticks([200, 400, 600], minor=True)
        for item in (
            [ax.title, ax.xaxis.label, ax.yaxis.label]
            + ax.get_xticklabels()
            + ax.get_yticklabels()
        ):
            item.set_fontsize(9)

    plt.tight_layout()
    plt.show()
    fig.savefig("images/samplesize_vs_auc.png")


make_plot()
