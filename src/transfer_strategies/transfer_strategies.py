from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, List, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb

from src.domains.domains import Domain
from src.xgb_tuning.xgb_base_training import (
    ModelEvaluation,
    continue_fit,
    get_model_evaluation,
    prune_trees,
    reweight_leaves,
    simple_fit,
    tune,
)


@dataclass
class SamplingTask:
    n: int
    sample_seed: int


@dataclass
class SimulationSetUp(ABC):
    source_domain: Domain
    target_domain: Domain
    sampling_task_source_train: SamplingTask
    sampling_task_target_train: SamplingTask
    sampling_task_source_test: SamplingTask = field(init=False)
    sampling_task_target_test: SamplingTask = field(init=False)

    def __post_init__(self):
        self.sampling_task_source_test = self.get_test_task(
            self.sampling_task_source_train
        )
        self.sampling_task_target_test = self.get_test_task(
            self.sampling_task_target_train
        )

    def get_test_task(self, train_task: SamplingTask):
        """ "
        Generates a matching test SamplingTask for a given training SamplingTask.

        Args:
            train_task (SamplingTask): the task that generates the training data

        Returns:
            SamplingTask:   A derived SamplingTask that can generates the test data.
                            Size of test data is always 1000. Sample seed is fixed
                            but different from training sample seed"
        """
        return SamplingTask(n=1000, sample_seed=train_task.sample_seed + 1)


@dataclass
class StrategyExecutionResult:
    simulation_set_up: SimulationSetUp | None
    strategy: str | None
    model_evaluation_on_src: ModelEvaluation | None = field(kw_only=True)
    model_evaluation_on_tgt: ModelEvaluation | None = field(kw_only=True)


class AbstractTransferStrategy(ABC):
    """
    Abstract transfer strategy
    """

    def __init__(self) -> None:
        self.name = self.__class__.__name__

    def sample_from_source_train(
        self, simulation_setup: SimulationSetUp
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return self.sample(
            domain=simulation_setup.source_domain,
            sampling_task=simulation_setup.sampling_task_source_train,
        )

    def sample_from_source_test(
        self, simulation_setup: SimulationSetUp
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return self.sample(
            domain=simulation_setup.source_domain,
            sampling_task=simulation_setup.sampling_task_source_test,
        )

    def sample_from_target_train(
        self, simulation_setup: SimulationSetUp
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return self.sample(
            domain=simulation_setup.target_domain,
            sampling_task=simulation_setup.sampling_task_target_train,
        )

    def sample_from_target_test(
        self, sampling_setup: SimulationSetUp
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return self.sample(
            domain=sampling_setup.target_domain,
            sampling_task=sampling_setup.sampling_task_target_test,
        )

    def sample(
        self, domain: Domain, sampling_task: SamplingTask
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return domain.generate_data(
            n=sampling_task.n, sample_seed=sampling_task.sample_seed
        )

    def get_summary(
        self,
        model_eval_on_src: ModelEvaluation | None,
        model_eval_on_tgt: ModelEvaluation | None,
        simulation_setup: SimulationSetUp,
    ) -> StrategyExecutionResult:
        """_summary_

        Args:
            model_eval_on_src (ModelEvaluation | None): Model evaluation on oos source data
            model_eval_on_tgt (ModelEvaluation | None): Model evaluation on oos target data
            simulation_setup (SimulationSetUp): Simulation set-up with sampling seeds and sample sizes

        Returns:
            SimulationSummary: dataclass with the bundled information
        """
        return StrategyExecutionResult(
            strategy=self.name,
            model_evaluation_on_src=model_eval_on_src,
            model_evaluation_on_tgt=model_eval_on_tgt,
            simulation_set_up=simulation_setup,
        )

    def get_best_model_evaluation(
        self, list_of_models: List[ModelEvaluation]
    ) -> ModelEvaluation | None:
        """
        Compares the AUC of different ModelEvaluations and
        returns the ModelEvaluation with the highest AUC

        Args:
            list_of_models (List[ModelEvaluation]): ModelEvaluation to compare

        Returns:
            ModelEvaluation|None: ModelEvaluation with the highest AUC
        """
        best_AUC = 0
        best_model = None
        for m in list_of_models:
            if m.auc_score > best_AUC:
                best_model = m
        return best_model

    def merge_datasets(
        self,
        src_X: pd.DataFrame,
        src_y: pd.DataFrame,
        tgt_X: pd.DataFrame,
        tgt_y: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        merged_X = pd.concat(src_X, tgt_X)
        merged_y = pd.concat(src_y, tgt_y)
        return merged_X, merged_y

    @abstractmethod
    def execute_strategy(
        self, simulation_setup: SimulationSetUp
    ) -> StrategyExecutionResult:
        ...


class SRCONLY(AbstractTransferStrategy):
    "trains a model on the source data only. Evaluates the source model on the target data"

    def execute_strategy(
        self, simulation_setup: SimulationSetUp
    ) -> StrategyExecutionResult:
        src_X_train, src_y_train = self.sample_from_source_train(simulation_setup)
        tuning_result = tune(src_X_train, y_train=src_y_train)

        src_X_test, src_y_test = self.sample_from_source_test(simulation_setup)
        src_evaluation = get_model_evaluation(
            tuning_output=tuning_result, X_test=src_X_test, y_test=src_y_test
        )

        tgt_X_test, tgt_y_test = self.sample_from_target_test(simulation_setup)
        tgt_evaluation = get_model_evaluation(
            tuning_output=tuning_result, X_test=tgt_X_test, y_test=tgt_y_test
        )

        summary = self.get_summary(src_evaluation, tgt_evaluation, simulation_setup)

        return summary


class TGTONLY(AbstractTransferStrategy):
    """trains a model on the source domain only. Evaluates the target model on the target data"""

    def execute_strategy(
        self, simulation_setup: SimulationSetUp
    ) -> StrategyExecutionResult:
        tgt_X_train, tgt_y_train = self.sample_from_target_train(simulation_setup)
        tgt_X_test, tgt_y_test = self.sample_from_target_test(simulation_setup)

        tuning_result = tune(X_train=tgt_X_train, y_train=tgt_y_train)

        tgt_evaluation = get_model_evaluation(
            tuning_output=tuning_result, X_test=tgt_X_test, y_test=tgt_y_test
        )

        summary = self.get_summary(
            model_eval_on_src=None,
            model_eval_on_tgt=tgt_evaluation,
            simulation_setup=simulation_setup,
        )

        return summary


@dataclass
class SimulationSummary_WEIGHTED(StrategyExecutionResult):
    simulation_setup: SimulationSetUp
    model_evaluation_on_tgt: ModelEvaluation


class WEIGHTED(AbstractTransferStrategy):
    def get_weight_vector(self, src_y: pd.DataFrame, tgt_y: pd.DataFrame):
        src_n = src_y.shape[1]
        tgt_n = tgt_y.shape[1]
        src_weights = [tgt_n / src_n for _ in range(src_n)]
        tgt_weights = [1.0 for _ in range(tgt_n)]
        all_weights = src_weights.append(tgt_weights)
        return all_weights

    def execute_strategy(
        self, simulation_setup: SimulationSetUp
    ) -> StrategyExecutionResult:
        src_X_train, src_y_train = self.sample_from_source_train(simulation_setup)
        tgt_X_train, tgt_y_train = self.sample_from_target_train(simulation_setup)
        merged_X_train, merged_y_train = self.merge_datasets(
            src_X_train, src_y_train, tgt_X_train, tgt_y_train
        )
        weights = self.get_weight_vector(src_y_train, tgt_y_train)

        tuning = tune(
            X_train=merged_X_train, y_train=merged_y_train, sample_weights=weights
        )

        tgt_X_test, tgt_y_test = self.sample_from_target_test(simulation_setup)
        tgt_evaluation = get_model_evaluation(
            tuning_output=tuning, X_test=tgt_X_test, y_test=tgt_y_test
        )

        summary = self.get_summary(
            model_eval_on_src=None,
            model_eval_on_tgt=tgt_evaluation,
            simulation_setup=simulation_setup,
        )
        return summary


class CONTINUE(AbstractTransferStrategy):
    def execute_strategy(
        self, simulation_setup: SimulationSetUp
    ) -> StrategyExecutionResult:
        src_X_train, src_y_train = self.sample_from_source_train(simulation_setup)
        tgt_X_train, tgt_y_train = self.sample_from_target_train(simulation_setup)
        tgt_X_test, tgt_y_test = self.sample_from_target_test(
            sampling_setup=simulation_setup
        )

        # train a fully tuned model on source data
        src_only_result = SRCONLY().execute_strategy(simulation_setup=simulation_setup)
        n_src_trees, src_model, src_params = self.get_base_tuning_output(
            src_only_result
        )

        n_src_trees = self.get_head_model_n_estimators(n_src_trees)

        candidate_models = []
        for rate, n in n_src_trees.items():
            # keep src_model parameters except n_estimators
            head_model = self.get_head_model(params=src_params, n_estimators=n)
            # fit model with n estimators on scr data
            head_model = simple_fit(src_X_train, src_y_train, head_model).booster
            # continue fitting with tgt data
            full_model = continue_fit(tgt_X_train, tgt_y_train, head_model, src_params)
            full_model_eval = get_model_evaluation(full_model, tgt_X_test, tgt_y_test, key = rate)
            candidate_models.append(full_model_eval)

        best_model_eval = self.get_best_model_evaluation(candidate_models)
        summary = self.get_summary(
            model_eval_on_src=None,
            model_eval_on_tgt=best_model_eval,
            simulation_setup=simulation_setup,
        )
        return summary

    def get_head_model_n_estimators(self, n_src_trees: int) -> list[int]:
        keep_rates = [0.2, 0.4, 0.6, 0.8]
        n = [round(n_src_trees * rate) for rate in keep_rates]
        return dict(zip(keep_rates, n))

    def get_base_tuning_output(
        self, base_execution_result: StrategyExecutionResult
    ) -> Tuple[Any]:
        model_0 = base_execution_result.model_evaluation_on_src.tuning_output.booster
        params_0 = (
            base_execution_result.model_evaluation_on_src.tuning_output.best_params
        )
        n_trees_0 = params_0["n_estimators"]
        return n_trees_0, model_0, params_0

    def get_head_model(self, params: dict, n_estimators) -> xgb.XGBClassifier:
        """_summary_

        Args:
            params (dict): Parameters obtained from training source only model
            n_estimators (_type_): number of trees to keep from source model

        Returns:
            xgb.XGBClassifier: Source model with fewer trees
        """
        params["n_estimators"] = n_estimators
        return xgb.XGBClassifier(**params)


class REVISE(AbstractTransferStrategy):
    def execute_strategy(
        self, simulation_setup: SimulationSetUp
    ) -> StrategyExecutionResult:
        src_X_train, src_y_train = self.sample_from_source_train(simulation_setup)
        tgt_X_train, tgt_y_train = self.sample_from_target_train(simulation_setup)
        tgt_X_test, tgt_y_test = self.sample_from_target_test(simulation_setup)

        # train a fully tuned model on source data
        src_only_result = SRCONLY().execute_strategy(simulation_setup=simulation_setup)
        src_model = src_only_result.model_evaluation_on_src.tuning_output.booster
        src_model_eval = get_model_evaluation(
            src_only_result.model_evaluation_on_tgt.tuning_output,
            X_test=tgt_X_test,
            y_test=tgt_y_test,
            key="src",
        )

        min_population = xgb.XGBClassifier(booster=src_model).get_params()[
            "min_child_weight"
        ]

        # fill sparse leaves in tgt data with src data
        X_fill_up, y_fill_up = self.fill_up_training_data(
            src_X_train=src_X_train,
            src_y_train=src_y_train,
            tgt_X_train=tgt_X_train,
            src_model=src_model,
            min_population=min_population,
        )
        merged_X, merged_y = self.merge_datasets(
            X_fill_up, y_fill_up, tgt_X_train, tgt_y_train
        )

        # Decide if reweight, prune or both should be used
        # reweight only
        reweighted = reweight_leaves(
            src_model=src_model, merged_X=merged_X, merged_y=merged_y
        )
        reweighted_eval = get_model_evaluation(
            tuning_output=reweighted,
            X_test=tgt_X_test,
            y_test=tgt_y_test,
            key="reweighted_only",
        )

        # prune only
        pruned = prune_trees(src_model, merged_X=merged_X, merged_y=merged_y)
        pruned_eval = get_model_evaluation(
            pruned, X_test=tgt_X_test, y_test=tgt_y_test, key="pruned_only"
        )

        # reweight AND prune
        reweighted_pruned = prune_trees(
            reweighted.booster, merged_X=merged_X, merged_y=merged_y
        )
        reweighted_pruned_eval = get_model_evaluation(
            reweighted_pruned,
            X_test=tgt_X_test,
            y_test=tgt_y_test,
            key="reweighted_and_pruned",
        )
        best_model = self.get_best_model_evaluation(
            [reweighted_eval, pruned_eval, reweighted_pruned_eval, src_model_eval]
        )

        summary = self.get_summary(
            model_eval_on_tgt=best_model,
            model_eval_on_src=None,
            simulation_setup=simulation_setup,
        )
        return summary

    def fill_up_training_data(
        self, src_X_train, src_y_train, tgt_X_train, src_model, min_population
    ):
        tgt_leaf_assignments = self.assign_leaves_to_samples(
            X=tgt_X_train, model=src_model
        )
        leaf_population = self.get_leaf_population(
            X=tgt_leaf_assignments, model=src_model
        )
        sparse_leaves = self.get_sparse_leaves(
            leaf_population, min_population=min_population
        )

        # fill-up with source_data
        X_fill_up, y_fill_up = self.get_fill_up_samples(
            src_X_train=src_X_train,
            src_y_train=src_y_train,
            sparse_leaves=sparse_leaves,
            model=src_model,
        )

        return X_fill_up, y_fill_up

    def get_fill_up_samples(
        self,
        src_X_train: pd.DataFrame,
        src_y_train: pd.DataFrame,
        sparse_leaves: pd.DataFrame,
        model: xgb.XGBClassifier,
    ):
        src_leaf_assignments = self.assign_leaves_to_samples(X=src_X_train, model=model)

        X_fill_up = pd.DataFrame(columns=src_X_train.columns)
        y_fill_up = pd.DataFrame(columns=src_y_train.columns)

        for row in sparse_leaves.itertuples():
            tree = row[0][0]
            leaf = row[0][1]
            deficit = row.deficit

            eligibile_indices = src_leaf_assignments[
                (src_leaf_assignments["tree"] == tree)
                & (src_leaf_assignments["leaf"] == leaf)
            ]
            random_samples = eligibile_indices.sample(
                n=deficit, random_state=deficit, replace=True
            )

            X_random_samples = src_X_train[
                src_X_train.index.isin(eligibile_indices.index)
            ]
            y_random_samples = src_y_train[
                src_X_train.index.isin(eligibile_indices.index)
            ]

            X_fill_up = X_fill_up.append(X_random_samples)
            y_fill_up = y_fill_up.append(y_random_samples)

        return X_fill_up, y_fill_up

    def assign_leaves_to_samples(
        self, X: pd.DataFrame, model: xgb.Booster
    ) -> pd.DataFrame:
        leaf_assignments = (
            # pd.DataFrame(model.apply(X))
            pd.DataFrame(model.predict(xgb.DMatrix(X), pred_leaf=True))
            .rename_axis("sample_id")
            .reset_index()
            .melt(id_vars=["sample_id"], var_name="tree", value_name="leaf")
            .sort_values(["sample_id", "tree", "leaf"])
        )
        # leaf_assignments = model.predict(xgb.DMatrix(X), pred_leaf=True)
        return leaf_assignments

    def get_leaf_population(self, X: pd.DataFrame, model: xgb.XGBClassifier):
        # Apply the classifier to get the leaf indices
        population_per_leaf = (
            X.groupby(by=["tree", "leaf"])
            .count()
            .rename(columns={"sample_id": "population_count"})
        )

        # leaf_indices = pd.wide_to_long(leaf_indices,stubnames = ['tree_i', 'leaf_j'], i = [leaf_indices.index, leaf_indices.columns])

        # Print the count of samples per leaf
        return population_per_leaf

    def get_sparse_leaves(self, X: pd.DataFrame, min_population: int):
        sparse_leaves = X[X["population_count"] < min_population]
        sparse_leaves["deficit"] = min_population - sparse_leaves["population_count"]
        return sparse_leaves

    def merge_datasets(
        self,
        src_X: pd.DataFrame,
        src_y: pd.DataFrame,
        tgt_X: pd.DataFrame,
        tgt_y: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        merged_X = pd.concat([src_X, tgt_X])
        merged_y = pd.concat([src_y, tgt_y])
        return merged_X, merged_y


cont = CONTINUE()

src_dom = Domain(567, 5)
tgt_dom = Domain(101, 5)
src_sampling = SamplingTask(n=1000, sample_seed=123)
tgt_sampling = SamplingTask(n=1230, sample_seed=123)
setup = SimulationSetUp(tgt_dom, src_dom, src_sampling, tgt_sampling)
summary = cont.execute_strategy(simulation_setup=setup)
print(summary)
