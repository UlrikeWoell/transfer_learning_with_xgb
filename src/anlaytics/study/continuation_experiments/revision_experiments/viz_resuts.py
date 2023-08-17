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

    def continuation_result_to_row(self, scenario: str, experiment: str):
        data = self.read_json(
            f"{self.root_dir}/{scenario}/{experiment}/continuation_result.json"
        )
        row = {
            "strategy": "Continuation",
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

    def continuation_es_result_to_row(self, scenario: str, experiment: str):
        data = self.read_json(
            f"{self.root_dir}/{scenario}/{experiment}/continuation_es_result.json"
        )
        row = {
            "strategy": "Continuation ES",
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

    def collect_continuation_variants_experiment(self, scenario: str, experiment: str):
        earlystopping = self.continuation_es_result_to_row(scenario, experiment)
        continuation = self.continuation_result_to_row(scenario, experiment)
        return [earlystopping, continuation]

    def collect_all_experiments(self):
        data = []
        scenarios = os.listdir(self.root_dir)
        for s in scenarios:
            experiments = os.listdir(f"{self.root_dir}/{s}")
            for e in experiments:
                data = data + self.collect_continuation_variants_experiment(s, e)
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
                x="auc_roc",
                y="experiment",
                hue="strategy",
                data=subset,
                ax=ax,
                jitter=False,
                dodge=True,
                marker="o",
            )

            ax.set_ylabel(f"Scenario:\n{scenario}")
            ax.axes.yaxis.set_ticklabels([])
            ax.grid(axis="y")
            ax.set_xlabel("AUC-ROC" if i == 4 else "")
            ax.set_xlim(0, 1)
            if i == 0:
                ax.legend(title="Strategy")
            else:
                ax.legend().remove()

        plt.tight_layout()
        plt.show()


data = DataCollector().collect_all_experiments()
print(data)

Plotter().make_stripplots(data)
