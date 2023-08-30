import json
import os
from abc import ABC, abstractmethod
from pprint import pprint
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pandas.plotting import parallel_coordinates
from PIL import Image


class Plot(ABC):
    def normalized_gain(self, initial_val: float, final_val: float):
        if not final_val or not initial_val:
            return None
        if final_val > initial_val:
            return (final_val - initial_val) / (1 - initial_val)
        else:
            return (final_val - initial_val) / (initial_val - 0.5)

    def __init__(self) -> None:
        self.root_dir = "data/simulation"

    def read_json(self, file: str):
        try:
            with open(file) as fp:
                data = json.load(fp)
            return data
        except:
            print(f"No such file: {file}")

    def collect_all_experiments(self):
        data = []
        scenarios = os.listdir(self.root_dir)
        for s in scenarios:
            experiments = os.listdir(f"{self.root_dir}/{s}")
            for e in experiments:
                data.append(self.collect_data_from_experiment(s, e))
        return pd.DataFrame(data)

    def collect_data_from_experiment(self, scenario: str, experiment: str):
        print(scenario, experiment)
        result_files = [
            "tgtonly_result.json",
            "srconly_result.json",
            "combination_result.json",
            "freeze_result.json",
            "progressive_result.json",
            "finetuning_result_.json",
            "finetuning_result_a.json",
            "finetuning_result_p.json",
            "finetuning_result_r.json",
            "finetuning_result_ap.json",
            "finetuning_result_ar.json",
            "finetuning_result_pr.json",
            "finetuning_result_apr.json",
        ]
        strategies = [
            "Target only",
            "Source only",
            "Combination",
            "Freeze",
            "Progressive learning",
            "Finetune x",
            "Finetune A",
            "Finetune P",
            "Finetune R",
            "Finetune AP",
            "Finetune AR",
            "Finetune PR",
            "Finetune APR",
        ]
        row = {
            "scenario": scenario,
            "experiment": experiment,
        }

        baseline = self.read_json(
            f"{self.root_dir}/{scenario}/{experiment}/{'tgtonly_result.json'}"
        )
        if baseline and baseline["success"]:
            initial_roc = baseline["auc_roc"] * 1.0
        else:
            return row


        for file, strategy in zip(result_files, strategies):
            data = self.read_json(f"{self.root_dir}/{scenario}/{experiment}/{file}")
            if data and data["success"]:
                final_roc = data["auc_roc"] * 1.0
                row.update(
                    {f"{strategy}": self.normalized_gain(initial_roc, final_roc)}
                )

        return row

    def make_plot(self):
        df = self.collect_all_experiments()
        print(df.describe())
        scenarios = [
            "bias",
            "censor",
            "coeffs_any_sign",
            "coeffs_same_sign",
            "exponents",
            "matrix_any_sign",
            "matrix_same_sign",
        ]

        sns.set_theme()
        for i, scenario in enumerate(scenarios):
            axes = 2
            fig, axes = plt.subplots(figsize=(8, 4), sharex=False, sharey=True)
            # ax = axes[i]
            subset = df[(df["scenario"] == scenario)]
            subset = subset.drop("Target only", axis=1)
            sns.swarmplot(subset, orient="h", size=3)
            plt.axvline(x=0, linestyle="dashed", color="gray")
            plt.title(scenario, fontsize=9)
            plt.xlabel("normalized gain/loss", fontsize=9)
            plt.ylabel("Transfer strategy", fontsize=9)
            for item in (
                [axes.xaxis.label, axes.yaxis.label]
                + axes.get_xticklabels()
                + axes.get_yticklabels()
            ):
                item.set_fontsize(9)

            plt.tight_layout()
            plt.show()
            fig.savefig(f"images/gain_dots_{scenario}.png")


p = Plot()
p.make_plot()
