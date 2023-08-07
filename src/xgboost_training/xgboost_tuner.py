import json
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import pandas as pd
import xgboost as xgb
from scipy.stats import uniform
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV


@dataclass
class XGBoostTunerConfig:
    random_seed = 27071990
    learning_rate_range = (0.025, 0.02)
    gamma_range = (0.2, 0.3)
    max_depth_values = [2, 3, 5, 7, 10, 30]
    colsample_bylevel = 0.5
    subsample = 0.75
    n_iterations = 50
    cv = 2


class XGBoostTuner:
    def __init__(self, tuner_config: XGBoostTunerConfig):
        self.random_seed = tuner_config.random_seed
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
    ) -> Dict[str, Any]:
        model = xgb.XGBClassifier(
            random_state=self.random_seed,
            colsample_bylevel=self.colsample_bylevel,
            subsample=self.subsample,
        )

        param_dist = {
            "learning_rate": uniform(*self.learning_rate_range),
            "gamma": uniform(*self.gamma_range),
            "max_depth": self.max_depth_values,
        }

        random_search = RandomizedSearchCV(
            model,
            param_distributions=param_dist,
            n_iter=self.n_iterations,
            scoring="roc_auc",
            random_state=self.random_seed,
            cv=self.cv,
            n_jobs=-1,
        )

        random_search.fit(X_train, y_train)

        tuning_results = {
            "success": True,
            "msg": None,
            "learning_rate": random_search.best_params_["learning_rate"],
            "gamma": random_search.best_params_["gamma"],
            "max_depth": random_search.best_params_["max_depth"],
        }

        metrics = self.calculate_metrics(random_search.best_estimator_, X_test, y_test)
        tuning_results.update(metrics)

        return tuning_results

    def calculate_metrics(
        self, model: xgb.XGBClassifier, X_test: pd.DataFrame, y_test: pd.DataFrame
    ) -> Dict[str, float]:
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)

        auc_roc = roc_auc_score(y_test, y_pred_proba)
        auc_pr = average_precision_score(y_test, y_pred_proba)
        f1 = f1_score(y_test, y_pred)

        return {"auc_roc": auc_roc, "auc_pr": auc_pr, "f1": f1}

    def save_results_to_json(
        self, file_path: str, file_name: str, tuning_results: Dict[str, Any]
    ):
        result_file = f"{file_path}/{file_name}"
        with open(result_file, "w") as f:
            json.dump(tuning_results, f)

    def save_failed_result_to_json(self, file_path: str, file_name: str, msg:str|None=None):
        tuning_result = {'success':False,
                         'msg':msg}
        self.save_results_to_json(file_path=file_path,
                                  file_name=file_name,
                                  tuning_results=tuning_result)
