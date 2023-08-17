import json
import os
from abc import ABC, abstractmethod
from pprint import pprint
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from PIL import Image


class Plot(ABC):
    def __init__(self) -> None:
        self.root_dir = "data/simulation/"

    def read_json(self, file: str):
        try:
            with open(file) as fp:
                data = json.load(fp)
            return data
        except:
            print(f"No such file: {file}")


class StripPlot(Plot):
    def srconly_result_to_row(self, scenario: str, experiment: str):
        data = self.read_json(
            f"{self.root_dir}/{scenario}/{experiment}/srconly_result.json"
        )
        row = {
            "strategy": "Source only",
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

    def tgtonly_result_to_row(self, scenario, experiment):
        data = self.read_json(
            f"{self.root_dir}/{scenario}/{experiment}/tgtonly_result.json"
        )
        row = {
            "strategy": "Target only",
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

    def combination_result_to_row(self, scenario: str, experiment: str):
        data = self.read_json(
            f"{self.root_dir}/{scenario}/{experiment}/combination_result.json"
        )
        row = {
            "strategy": "Combination",
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

    def revision_result_to_row(self, scenario: str, experiment: str):
        data = self.read_json(
            f"{self.root_dir}/{scenario}/{experiment}/revision_result.json"
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

    def collect_data_from_experiment(self, scenario: str, experiment: str):
        tgt_only = self.tgtonly_result_to_row(scenario, experiment)
        src_only = self.srconly_result_to_row(scenario, experiment)
        combination = self.combination_result_to_row(scenario, experiment)
        continuation = self.continuation_es_result_to_row(scenario, experiment)
        revsion = self.revision_result_to_row(scenario, experiment)
        return [tgt_only, src_only, combination, continuation, revsion]

    def collect_all_experiments(self):
        data = []
        scenarios = os.listdir(self.root_dir)
        for s in scenarios:
            experiments = os.listdir(f"{self.root_dir}/{s}")
            for e in experiments:
                data = data + self.collect_data_from_experiment(s, e)
        return pd.DataFrame(data)

    def make_stripplots(self):
        df = self.collect_all_experiments()

        sns.set_theme()
        fig, axes = plt.subplots(5, figsize=(8, 8))
        scenarios = ["bias", "censor", "exponents", "matrix", "coeffs"]
        for i, scenario in enumerate(scenarios):
            ax = axes[i]
            subset = df[(df["scenario"] == scenario)]
            g = sns.stripplot(
                y="auc_roc",
                x="experiment",
                hue="strategy",
                data=subset,
                ax=ax,
                jitter=False,
                dodge=True,
                marker="o",
            )

            ax.set_ylabel(f"Scenario:\n{scenario}")
            ax.axes.xaxis.set_ticklabels([])
            ax.grid(axis="x")
            ax.set_xlabel("AUC-ROC" if i == 4 else "")
            ax.set_ylim(0, 1.1)
            if i == 0:
                ax.legend(title="Strategy")
            else:
                ax.legend().remove()

        plt.tight_layout()
        plt.show()


class ParaCoordPlot(Plot):
    def get_data(self, scenario: str, experiment: str, filename: str):
        data = self.read_json(f"{self.root_dir}/{scenario}/{experiment}/{filename}")
        if data and data["success"]:
            return data["auc_roc"]

        else:
            return None

    def collect_experiment(self, scenario: str, experiment: str):
        files = [
            ("Source only", "srconly_result.json"),
            ("Target only", "tgtonly_result.json"),
            ("Combination", "combination_result.json"),
            ("Continuation", "continuation_result.json"),
            ("Revision", "revision_result.json"),
        ]

        row = {
            "scenario": scenario,
            "experiment": experiment,
        }
        for label, filename in files:
            row.update({label: self.get_data(scenario, experiment, filename)})
        return row

    def collect_all_experiments(self):
        data = []
        scenarios = os.listdir(self.root_dir)
        for s in scenarios:
            experiments = os.listdir(f"{self.root_dir}/{s}")
            for e in experiments:
                data.append(self.collect_experiment(s, e))
        return pd.DataFrame(data)

    def parallel_coordinates(self):
        df = self.collect_all_experiments()
        sns.set_theme()
        fig, axes = plt.subplots(5, figsize=(8, 8))
        plt.title("AUC-ROC for different experiments")
        scenarios = ["bias", "censor", "exponents", "matrix", "coeffs"]
        for i, scenario in enumerate(scenarios):
            ax = axes[i]
            subset = df[(df["scenario"] == scenario)]
        
            pd.plotting.parallel_coordinates(
                subset,
                "scenario",
                cols=[
                    "Source only",
                    "Target only",
                    "Combination",
                    "Continuation", 
                    "Revision",
                ],
                color=["lime", "tomato", "dodgerblue", "green"],
                alpha=0.2,
                axvlines_kwds={"color": "black"},
                ax=ax
            )
            ax.set_ylabel(f"Scenario:\n{scenario}")
            ax.axes.xaxis.set_ticklabels([])
            ax.set_ylim(0, 1.1)
            ax.legend().remove()

        plt.tight_layout()
        plt.show()


ParaCoordPlot().parallel_coordinates()

