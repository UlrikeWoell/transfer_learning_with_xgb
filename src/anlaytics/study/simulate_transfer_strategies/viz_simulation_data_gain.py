import json
import os
from abc import ABC, abstractmethod
from pprint import pprint
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
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
        self.strategies = [
            "Target only",
            "Source only",
            "Combination",
            "Freeze",
            "Progressive learning",
            # "Finetune x",
            # "Finetune A",
            "Finetune P",
            "Finetune R",
            "Finetune PR",
            "Finetune AP",
            "Finetune AR",
            "Finetune APR",
        ]

        self.scenarios = [
            "exponents",
            "matrix_same_sign",
            "matrix_any_sign",
            "bias",
            "coeffs_same_sign",
            "coeffs_any_sign",
            "censor",
        ]

        self.scenario_titles = [
            "UnivShape",
            "CorrSame",
            "CorrAny",
            "ClassRatio",
            "CoeffSame",
            "CoeffAny",
            "FeatOverlap",
        ]

        self.scenario_map = {}
        for short, long in zip(self.scenarios, self.scenario_titles):
            self.scenario_map.update({short: long})

    def read_json(self, file: str):
        try:
            with open(file) as fp:
                data = json.load(fp)
            return data
        except:
            print(f"No such file: {file}")

    def collect_all_experiments(self, ranks: bool = False):
        data = []
        scenarios = os.listdir(self.root_dir)
        for s in scenarios:
            experiments = os.listdir(f"{self.root_dir}/{s}")
            for e in experiments:
                if ranks:
                    data.append(self.collect_ranks_from_experiment(s, e))
                else:
                    data.append(self.collect_data_from_experiment(s, e))
        return pd.DataFrame(data)

    def collect_data_from_experiment(self, scenario: str, experiment: str):
        result_files = [
            "tgtonly_result.json",
            "srconly_result.json",
            "combination_result.json",
            "freeze_result.json",
            "progressive_result.json",
            # "finetuning_result_.json",
            # "finetuning_result_a.json",
            "finetuning_result_p.json",
            "finetuning_result_r.json",
            "finetuning_result_ap.json",
            "finetuning_result_ar.json",
            "finetuning_result_pr.json",
            "finetuning_result_apr.json",
        ]

        row = {
            "scenario": scenario,
            "experiment": experiment,
        }

        baseline = self.read_json(
            f"{self.root_dir}/{scenario}/{experiment}/{'tgtonly_result.json'}"
        )
        if baseline and baseline["success"]:
            initial_roc = baseline["auc_pr"] * 1.0
        else:
            return row

        for file, strategy in zip(result_files, self.strategies):
            data = self.read_json(f"{self.root_dir}/{scenario}/{experiment}/{file}")
            if data and data["success"]:
                final_roc = data["auc_pr"] * 1.0
                row.update(
                    {f"{strategy}": self.normalized_gain(initial_roc, final_roc)}
                )

        return row

    def collect_ranks_from_experiment(self, scenario: str, experiment: str):
        gain_row = self.collect_data_from_experiment(scenario, experiment)
        rank_row = {"experiment": experiment, "scenario": scenario}
        del gain_row["scenario"], gain_row["experiment"]
        rank_row.update(
            {
                key: rank
                for rank, key in enumerate(
                    sorted(gain_row, key=gain_row.get, reverse=True), 1
                )
            }
        )
        return rank_row

    def make_dotplot_by_scenario(self):
        df = self.collect_all_experiments()
        scenarios = [
            "censor",
            "bias",
            "coeffs_any_sign",
            "coeffs_same_sign",
            "exponents",
            "matrix_any_sign",
            "matrix_same_sign",
        ]

        scenario_titles = [
            "Feature overlap problem",
            "Inductive problem: Class ratio",
            "Inductive problem: Coefficients with any sign",
            "Inductive problem: Coefficients with same sign",
            "Non-inductive problem: Univariate distribution shape",
            "Non-inductive problem: Correlations with any sign",
            "Non-inductive problem: Correlations with same sign",
        ]

        sns.set_theme()
        for i, scenario in enumerate(scenarios):
            fig, axes = plt.subplots(figsize=(8, 4), sharex=False, sharey=True)
            subset = df[(df["scenario"] == scenario)]
            subset = subset.drop("Target only", axis=1)
            sns.swarmplot(subset, orient="h", size=2)
            plt.axvline(x=0, linestyle="dashed", color="gray")
            plt.title(scenario_titles[i], fontsize=9)
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

    def make_dotplot_by_strategy(self):
        df = self.collect_all_experiments()
        df["scenario_title"] = df["scenario"].map(self.scenario_map)

        sns.set_theme()
        stratgies = self.strategies
        stratgies.remove("Target only")
        for strategy in self.strategies:
            fig, axes = plt.subplots(figsize=(8, 3), sharex=False, sharey=True)
            sns.stripplot(
                data=df,
                y="scenario_title",
                x=strategy,
                hue="scenario_title",
                orient="h",
                size=2.5,
                order=self.scenario_titles,
            )
            axes.get_legend().remove()
            plt.axvline(x=0, linestyle="dashed", color="gray")
            plt.title(f"Strategy: {strategy}", fontsize=9)
            plt.xlabel("Normalized Gain/Loss in AUC vs Target Only", fontsize=9)
            plt.ylabel("Scenario", fontsize=9)
            for item in (
                [axes.xaxis.label, axes.yaxis.label]
                + axes.get_xticklabels()
                + axes.get_yticklabels()
            ):
                item.set_fontsize(9)

            plt.tight_layout()
            # plt.show()
            fig.savefig(f"images/dots_by_strategy/gain_dots_{strategy}.png")

    def make_rank_plot(self):
        df = self.collect_all_experiments(ranks=True)
        scenarios = [
            "censor",
            "bias",
            "coeffs_any_sign",
            "coeffs_same_sign",
            "exponents",
            "matrix_any_sign",
            "matrix_same_sign",
        ]

        scenario_titles = [
            "Feature overlap problem",
            "Inductive problem:\nClass ratio",
            "Inductive problem:\nCoefficients with any sign",
            "Inductive problem:\nCoefficients with same sign",
            "Non-inductive problem:\nUnivariate distribution shape",
            "Non-inductive problem:\nCorrelations with any sign",
            "Non-inductive problem:\nCorrelations with same sign",
        ]
        strategies = [
            "Target only",
            "Source only",
            "Combination",
            "Freeze",
            "Progressive learning",
            # "Finetune x",
            # "Finetune A",
            # "Finetune P",
            # "Finetune R",
            # "Finetune AP",
            # "Finetune AR",
            # "Finetune PR",
            # "Finetune APR",
        ]
        strategy_titles = [
            "Target only",
            "Source only",
            "Combination",
            "Freeze",
            "Progressive\nlearning",
            # "Finetune x",
            # "Finetune A",
            # "Finetune P",
            # "Finetune R",
            # "Finetune AP",
            # "Finetune AR",
            # "Finetune PR",
            # "Finetune APR",
        ]

        sns.set_theme()
        ncols = len(scenarios)
        nrows = len(strategies)
        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(1.3 * nrows, 1.5 * ncols),
            sharex=True,
            sharey=True,
        )
        for i, strat in enumerate(strategies):
            for j, scenario in enumerate(scenarios):
                subset = df[(df["scenario"] == scenario)]
                subset = subset.drop("scenario", axis=1)
                subset = subset.drop("experiment", axis=1)
                frequency = subset.apply(lambda x: x.value_counts()).fillna(0)
                frequency["rank"] = frequency.index
                frequency = frequency.melt(
                    id_vars="rank", var_name="strategy", value_name="count"
                )

                strat_freq = frequency[frequency["strategy"] == strat]
                ax = axes[i][j]
                ax.vlines(
                    x=strat_freq["rank"],
                    ymin=0,
                    ymax=strat_freq["count"],
                    color="darkblue",
                )
                ax.scatter(
                    x=strat_freq["rank"], y=strat_freq["count"], s=2, color="brown"
                )

                if i == 0:
                    ax.set_title(f"{scenario_titles[j]}", fontsize=8)
                else:
                    ax.set_title("", fontsize=9)

                if i == nrows - 1 and j == 3:
                    ax.set_xlabel(f"{'Rank'}", fontsize=9)
                else:
                    ax.set_xlabel(f"{''}")

                if j == 0:
                    ax.set_ylabel(
                        f"{strategy_titles[i]}",
                        fontsize=8,
                        rotation="horizontal",
                        ha="right",
                        va="center",
                    )
                else:
                    ax.set_ylabel("", rotation=0)

                for item in ax.get_xticklabels() + ax.get_yticklabels():
                    item.set_fontsize(5)
                    item.set_color("grey")
                ax.grid(visible=True, which="minor", axis="x")

        fig.suptitle(f"Ranks of strategies in different scenarios", fontsize=9)
        plt.tight_layout()
        # plt.show()
        fig.savefig(f"images/ranks/ranks.png")

    def get_heatmap_of_rankings_data(self):
        df = self.collect_all_experiments(ranks=True)
        heat = []
        for scenario in self.scenarios:
            strategies = self.strategies.copy()
            subset = df[df["scenario"] == scenario]
            for s in strategies:
                for l in strategies:
                    n_all = subset.shape[0]
                    count = subset[subset[s] < subset[l]].shape[0]
                    prob = None if s == l else round(count / n_all, 2)
                    row = {
                        "scenario": scenario,
                        "smaller": s,
                        "larger": l,
                        f"prob": prob,
                        'label': "" if not prob else str(prob)
                    }
                    heat.append(row)

        return pd.DataFrame(heat)

    def make_heatmap(self):
        df = self.get_heatmap_of_rankings_data()
        sns.set_theme()

        for scenario in self.scenarios:
            fig, ax = plt.subplots(figsize=(6, 6))
            subset = df[df["scenario"] == scenario]
            df_heatmap = subset.pivot_table(
                values="prob", index="smaller", columns="larger", aggfunc=np.mean
            )
            df_heatmap = df_heatmap[self.strategies]
            df_heatmap = df_heatmap.reindex(self.strategies)
            df_heatmap = df_heatmap.rename(
                index={"Progressive learning": "Progr. Learning"},
                columns={"Progressive learning": "Progr. Learning"},
            )

            ax = sns.heatmap(
                df_heatmap,
                annot=True,
                center=0.5,
                cmap="coolwarm_r",
                square=True,
                annot_kws={"fontsize": 8},
                cbar=False,
                linewidth=1,
                linecolor="w",
            )
            for item in (
                [ax.xaxis.label, ax.yaxis.label]
                + ax.get_xticklabels()
                + ax.get_yticklabels()
            ):
                item.set_fontsize(8)
            plt.title(f"Scenario: {self.scenario_map[scenario]}", fontsize=10)
            plt.ylabel("Probability that this strategy ranks better ...", fontsize=9)
            plt.xlabel("... than this strategy", fontsize=9)
            plt.tight_layout()
            # plt.show()
            fig.savefig(f"images/ranks/rank_{self.scenario_map[scenario]}.png")


p = Plot()
# p.make_dotplot_by_strategy()
# p.make_rank_plot()
p.make_heatmap()
