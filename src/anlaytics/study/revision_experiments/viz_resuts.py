import json
import os
from abc import ABC, abstractmethod
from pprint import pprint
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from PIL import Image


class DataCollector:
    def __init__(self) -> None:
        self.root_dir = "data/simulation/"

    def read_json(self, file: str):
        try:
            with open(file) as fp:
                data = json.load(fp)
            return data
        except:
            print(f"No such file: {file}")

    def finetuning_result_to_row(self, scenario: str, experiment: str):
        data = self.read_json(
            f"{self.root_dir}/{scenario}/{experiment}/finetuning_result.json"
        )
        row = {
            "strategy": "Revision",
            "scenario": scenario,
            "experiment": experiment,
        }
        if data and data["success"]:
            row.update(
                {
                    "auc_roc": data["auc_roc"],
                    "auc_pr": data["auc_pr"],
                    "f1": data["f1"],
                }
            )
        return row

    def collect_finetuning_apr_data(self, scenario: str, experiment: str):
        finetuning_options = [
            [False, False, False],
            [False, False, True],
            [False, True, False],
            [False, True, True],
            [True, False, False],
            [True, False, True],
            [True, True, False],
            [True, True, True],
        ]
        rows = []

        for ro in finetuning_options:
            filename = f"finetuning_result_a{str(ro[0])}_p{str(ro[1])}_r{str(ro[2])}.json"
            strategy = f"a{str(ro[0])}_p{str(ro[1])}_r{str(ro[2])}"

            data = self.read_json(f"{self.root_dir}/{scenario}/{experiment}/{filename}")
            row = {
                "strategy": strategy,
                "scenario": scenario,
                "experiment": experiment,
                "augment": ro[0],
                "prune": ro[1],
                "reweigh": ro[2],
                "exp_a_p": f"{experiment}{ro[0]}{ro[1]}",
                "exp_a_r": f"{experiment}{ro[0]}{ro[2]}",
                "exp_p_r": f"{experiment}{ro[2]}{ro[1]}",
            }
            if data and data["success"]:
                row.update(
                    {
                        "auc_roc": data["auc_roc"],
                        "auc_pr": data["auc_pr"],
                        "f1": data["f1"],
                    }
                )
            rows.append(row)
        return rows

    def collect_all_experiments_finetuning_apr(self):
        data = []
        scenarios = os.listdir(self.root_dir)
        for s in scenarios:
            experiments = os.listdir(f"{self.root_dir}/{s}")
            for e in experiments:
                data = data + self.collect_finetuning_apr_data(s, e)
        return pd.DataFrame(data)


class Plotter:
    def make_stripplots(self, df: pd.DataFrame):
        sns.set_theme()
        fig, axes = plt.subplots(5, figsize=(8, 8))
        scenarios = ["bias", "censor", "exponents", "matrix", "coeffs"]
        for i, scenario in enumerate(scenarios):
            ax = axes[i]
            subset = df[(df["scenario"] == scenario)]
            g = sns.stripplot(
                y="auc_roc",
                x="exp_a_p",
                hue="reweigh",
                data=subset,
                ax=ax,
                jitter=False,
                dodge=False,
                marker="o",
            )

            ax.set_xlabel(f"Scenario:\n{scenario}")
            ax.axes.xaxis.set_ticklabels([])
            ax.grid(axis="x")
            ax.set_ylabel("AUC-ROC" if i == 4 else "")
            ax.set_ylim(0, 1.1)
            if i == 0:
                ax.legend(title="Strategy")
            else:
                ax.legend().remove()

        plt.tight_layout()
        plt.show()


data = DataCollector().collect_all_experiments_finetuning_apr()
print(data)

Plotter().make_stripplots(data)
