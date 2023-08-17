import json
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import uniform
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV


@dataclass
class XGBoostTunerConfig:
    random_seed: int
    n_estimators: List[int]
    learning_rate_range: tuple[float, float]
    gamma_range: tuple[float, float]
    max_depth_values: List[int]
    colsample_bylevel: float
    subsample: float
    n_iterations: int
    cv: int


DefaultXGBoostTunerConfig = XGBoostTunerConfig(
    random_seed=27071990,
    n_estimators=[50, 100, 150, 200],
    learning_rate_range=(0.025, 0.3),
    gamma_range=(0.1, 0.5),
    max_depth_values=[2, 3, 5, 7, 10, 30],
    colsample_bylevel=0.5,
    subsample=0.75,
    n_iterations=70,
    cv=2,
)


class XGBoostTuner:
    def __init__(self, tuner_config: XGBoostTunerConfig):
        self.random_seed = tuner_config.random_seed
        self.n_estimators = tuner_config.n_estimators
        self.learning_rate_range = tuner_config.learning_rate_range
        self.gamma_range = tuner_config.gamma_range
        self.max_depth_values = tuner_config.max_depth_values
        self.colsample_bylevel = tuner_config.colsample_bylevel
        self.subsample = tuner_config.subsample
        self.n_iterations = tuner_config.n_iterations
        self.cv = 2

    def tune_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_test: pd.DataFrame,
        sample_weight: pd.DataFrame | None = None,
    ) -> Tuple[Dict[str, Any], xgb.XGBClassifier]:
        model = self.get_model()
        param_dist = self.get_param_dist()

        random_search = RandomizedSearchCV(
            model,
            param_distributions=param_dist,
            n_iter=self.n_iterations,
            scoring="roc_auc",
            random_state=self.random_seed,
            cv=self.cv,
            n_jobs=-1,
        )

        random_search.fit(X_train, y_train, sample_weight=sample_weight)
        model = random_search.best_estimator_

        metrics = self.calculate_metrics(random_search.best_estimator_, X_test, y_test)

        tuning_results = {
            "success": True,
            "msg": None,
            "learning_rate": random_search.best_params_["learning_rate"],
            "gamma": random_search.best_params_["gamma"],
            "max_depth": random_search.best_params_["max_depth"],
            "rs_best_params": random_search.best_params_,
            "best_estimator_params": random_search.best_estimator_.get_params(),
        }

        tuning_results.update(metrics)

        return tuning_results, model

    def get_param_dist(self):
        param_dist = {
            "learning_rate": uniform(*self.learning_rate_range),
            "gamma": uniform(*self.gamma_range),
            "max_depth": self.max_depth_values,
            "n_estimators": self.n_estimators,
        }

        return param_dist

    def get_model(self):
        model = xgb.XGBClassifier(
            random_state=self.random_seed,
            colsample_bylevel=self.colsample_bylevel,
            subsample=self.subsample,
            verbosity=0,
        )

        return model

    def tune_continuation_model(
        self,
        X_train_1: pd.DataFrame,
        y_train_1: pd.DataFrame,
        X_train_2: pd.DataFrame,
        y_train_2: pd.DataFrame,
        X_test: pd.DataFrame,
        y_test: pd.DataFrame,
    ):
        tuning_result, model = self.tune_model(
            X_train=X_train_1, y_train=y_train_1, X_test=X_test, y_test=y_test
        )
        best_params = tuning_result["best_estimator_params"]
        t0 = best_params["n_estimators"]

        swap_after_n_trees = [round(i * t0) for i in [0, 0.2, 0.4, 0.6, 0.8, 1]]
        best_score = 0
        best_swap_time = None
        best_metrics = None

        for swap_time in swap_after_n_trees:
            model = xgb.XGBClassifier(**best_params)
            model.set_params(n_estimators=swap_time)
            model.fit(X_train_1, y_train_1)
            model.set_params(n_estimators=t0)
            model.fit(X_train_2, y_train_2, xgb_model=model.get_booster())
            metrics = self.calculate_metrics(model, X_test, y_test)
            score = metrics["auc_pr"]
            if score > best_score:
                best_score = score
                best_swap_time = swap_time
                best_metrics = metrics

        tuning_results = {
            "success": True,
            "msg": None,
            "learning_rate": best_params["learning_rate"],
            "gamma": best_params["gamma"],
            "max_depth": best_params["max_depth"],
            "swap_after_n_trees": best_swap_time,
            "total_number_of_trees": t0,
        }

        tuning_results.update(best_metrics)

        return tuning_results

    def tune_continuation_model_early_stopping(
        self,
        X_train_1: pd.DataFrame,
        y_train_1: pd.DataFrame,
        X_train_2: pd.DataFrame,
        y_train_2: pd.DataFrame,
        X_test: pd.DataFrame,
        y_test: pd.DataFrame,
    ):
        tuning_result, _ = self.tune_model(
            X_train=X_train_1, y_train=y_train_1, X_test=X_test, y_test=y_test
        )
        best_params = tuning_result["best_estimator_params"]
        eval_set = [(X_train_1,y_train_1),(X_test, y_test)]

        model = xgb.XGBClassifier(**best_params)
        model.set_params(
            early_stopping_rounds=20,
            eval_metric="logloss",
        )

        model.fit(
            X_train_1,
            y_train_1,
            eval_set=eval_set,
            verbose = False
        )
        step1_parameters = model.get_params()
        new_eta = 0.75 * model.get_params()["learning_rate"]
        new_n_estimators = round(2 * model.get_params()["n_estimators"])
        model.set_params(learning_rate=new_eta, n_estimators=new_n_estimators)

        model.fit(
            X_train_2,
            y_train_2,
            eval_set=eval_set,
            xgb_model=model.get_booster(),
            verbose = False
        )
        step2_parameters = model.get_params()

        tuning_results = {
            "success": True,
            "msg": None,
            "step1_params": step1_parameters,
            "step2_params": step2_parameters,
        }

        metrics = self.calculate_metrics(model=model, X_test=X_test, y_test=y_test)
        tuning_results.update(metrics)

        return tuning_results

    def tune_revision_model(
        self,
        X_train_src: pd.DataFrame,
        y_train_src: pd.DataFrame,
        X_train_tgt: pd.DataFrame,
        y_train_tgt: pd.DataFrame,
        X_test_src: pd.DataFrame,
        y_test_src: pd.DataFrame,
        X_test_tgt: pd.DataFrame,
        y_test_tgt: pd.DataFrame,
        augment: bool,
        prune: bool,
        reweight: bool,
    ):
        base_tuning_result, base_model = self.tune_model(
            X_train=X_train_src,
            y_train=y_train_src,
            X_test=X_test_src,
            y_test=y_test_src,
        )

        if augment:
            augmented_X_train, augmented_y_train = self._augment_target_data(
                base_model,
                X_train_src,
                y_train_src,
                X_train_tgt,
                y_train_tgt,
            )
        else:
            augmented_X_train = X_train_src
            augmented_y_train = y_train_src

        if prune:
            pruning_result = self._update_model_with_prune(
                base_model=base_model,
                X_train_new=augmented_X_train,
                y_train_new=augmented_y_train,
                gamma=base_tuning_result["gamma"],
            )
            pruned_model = pruning_result["updated_model"]
            pruning_summary = pruning_result["pruning_summary"]
        else:
            pruned_model = base_model
            pruning_summary = None

        if reweight:
            reweighted_model = self._update_model_leaf_weights(
                base_model=pruned_model,
                X_train_new=augmented_X_train,
                y_train_new=augmented_y_train,
            )
        else:
            reweighted_model = pruned_model

        metrics = self.calculate_metrics(
            model=reweighted_model, X_test=X_test_tgt, y_test=y_test_tgt
        )
        tuning_results = {
            "success": True,
            "msg": None,
            "best_params": reweighted_model.get_params(),
            "pruning_summary": pruning_summary,
        }

        tuning_results.update(metrics)

        return tuning_results

    def _augment_target_data(
        self,
        model: xgb.XGBClassifier,
        X_train_src: pd.DataFrame,
        y_train_src: pd.DataFrame,
        X_train_tgt: pd.DataFrame,
        y_train_tgt: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Fills the target training data with samples from the source data
        to meet a minimum leaf count threshold.

        This function uses an XGBoost model trained with the given
        base_params to predict the leaf indices for the source and
        target training data. If any leaves in the target data have
        fewer than `min_leaf_cnt` samples, additional samples from
        the source data with the same leaf index are added to the target data.

        This augmented target data can be used to revise the model to better fit the target domain.

        Parameters:
            base_params (dict): The parameters for the XGBoost model.
            X_train_src (pd.DataFrame): The feature matrix for the source training data.
            y_train_src (pd.DataFrame): The labels for the source training data.
            X_train_tgt (pd.DataFrame): The feature matrix for the target training data.
            y_train_tgt (pd.DataFrame): The labels for the target training data.

        Returns:
            pd.DataFrame: The augmented feature matrix for the target training data.
            pd.DataFrame: The augmented labels for the target training data.
        """
        min_leaf_cnt = 20

        booster = model.get_booster()
        src_leaf_indices = booster.predict(
            xgb.DMatrix(X_train_src), pred_leaf=True
        ).astype(int)
        tgt_leaf_indices = booster.predict(
            xgb.DMatrix(X_train_tgt), pred_leaf=True
        ).astype(int)

        augmented = pd.concat([X_train_tgt, y_train_tgt], axis=1)
        src = pd.concat([X_train_src, y_train_src], axis=1)

        # Iterate through trees
        for tree_idx in range(tgt_leaf_indices.shape[1]):
            # Get leaf indices for the current tree
            tgt_leaf_indices_tree = tgt_leaf_indices[:, tree_idx]
            src_leaf_indices_tree = src_leaf_indices[:, tree_idx]

            # Get counts for the target leaf indices
            tgt_leaf_counts = np.bincount(tgt_leaf_indices_tree)

            # Iterate through the leaf counts for the current tree
            for leaf_idx, count in enumerate(tgt_leaf_counts):
                if count < min_leaf_cnt:
                    extra_samples_needed = min_leaf_cnt - count

                    if extra_samples_needed > 0:
                        selected_indices = src_leaf_indices[:, tree_idx] == leaf_idx
                        same_leaf = src[selected_indices]

                        if len(same_leaf) > 0:
                            sampled = same_leaf.sample(
                                n=extra_samples_needed,
                                replace=True,
                                random_state=leaf_idx,
                            )

                            augmented = pd.concat(
                                [augmented, sampled]
                            ).drop_duplicates()
        augmented_y = augmented["y"]
        augmented_X = augmented.drop("y", axis=1)

        return augmented_X, augmented_y

    def _update_model_with_prune(
        self,
        base_model: xgb.XGBClassifier,
        X_train_new: pd.DataFrame,
        y_train_new: pd.DataFrame,
        gamma: float,
    ):
        """
        Updates the given base model using the "prune" updater.

        This function trains the model on new data, pruning branches with minimal gain
        as defined by the gamma parameter.

        Parameters:
        base_model: The base XGBoost model to be updated.
        X_train_new (pd.DataFrame): The new feature matrix for training.
        y_train_new (pd.DataFrame): The new labels for training.
        gamma (float): Threshold for pruning. Splits with gain less than gamma are pruned.

        Returns:
            An updated XGBoost model trained with the "prune" updater.
            int: The number of prunings performed (difference in leaf count between base and updated model)
        """

        params = {
            "updater": "prune",
            "gamma": gamma,
            "process_type": "update",
        }

        train_matrix_new = xgb.DMatrix(X_train_new, label=y_train_new)
        base_leaves_count = sum(
            line.count("leaf") for line in base_model.get_booster().get_dump()
        )
        updated_booster = xgb.train(
            params, train_matrix_new, xgb_model=base_model.get_booster()
        )
        updated_model = xgb.XGBClassifier()
        updated_model._Booster = updated_booster

        updated_leaves_count = sum(
            line.count("leaf") for line in updated_booster.get_dump()
        )
        prunings_count = base_leaves_count - updated_leaves_count
        ratio_pruned_vs_base = prunings_count / base_leaves_count

        return {
            "updated_model": updated_model,
            "pruning_summary": {
                "base_leaves_count": base_leaves_count,
                "updated_leaves_count": updated_leaves_count,
                "prunings_count": prunings_count,
                "ratio_pruned_vs_base": ratio_pruned_vs_base,
            },
        }

    def _update_model_leaf_weights(
        self,
        base_model: xgb.XGBClassifier,
        X_train_new: pd.DataFrame,
        y_train_new: pd.DataFrame,
    ) -> xgb.Booster:
        """
        Recalculates the leaf weights of the given base model using new data.

        This function uses the "refresh" updater to update the leaf weights of the
        model without altering the structure of the trees, making it suitable for
        adapting the model to new data without retraining from scratch.

        Parameters:
            base_model: The base XGBoost model to be updated.
            X_train_new (pd.DataFrame): The new feature matrix for training.
            y_train_new (pd.DataFrame): The new labels for training.

        Returns:
            An updated XGBoost model with recalculated leaf weights.
        """
        params = {"process_type": "update", "updater": "refresh", "refresh_leaf": True}
        train_matrix_new = xgb.DMatrix(X_train_new, label=y_train_new)
        updated_booster = xgb.train(
            params, train_matrix_new, xgb_model=base_model.get_booster()
        )
        updated_model = xgb.XGBClassifier()
        updated_model._Booster = updated_booster

        return updated_model

    def calculate_metrics(
        self, model: xgb.XGBClassifier, X_test: pd.DataFrame, y_test: pd.DataFrame
    ) -> Dict[str, float]:
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)

        try:
            auc_roc = roc_auc_score(y_test, y_pred_proba)
        except:
            auc_roc = None
        try:
            auc_pr = average_precision_score(y_test, y_pred_proba)
        except:
            auc_pr = None
        try:
            f1 = f1_score(y_test, y_pred)
        except:
            f1 = None

        return {"auc_roc": auc_roc, "auc_pr": auc_pr, "f1": f1}

    def save_results_to_json(
        self, file_path: str, file_name: str, tuning_results: Dict[str, Any]
    ):
        result_file = f"{file_path}/{file_name}"
        with open(result_file, "w") as f:
            json.dump(tuning_results, f)

    def save_failed_result_to_json(
        self, file_path: str, file_name: str, msg: str | None = None
    ):
        tuning_result = {"success": False, "msg": msg}
        self.save_results_to_json(
            file_path=file_path, file_name=file_name, tuning_results=tuning_result
        )
