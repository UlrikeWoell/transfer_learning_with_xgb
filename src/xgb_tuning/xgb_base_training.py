from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List
import pandas as pd
import xgboost as xgb
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier

from src.util.config_reader import Configuration


def get_classifier(pre_trained_model: XGBClassifier | None):
    if pre_trained_model:
        return pre_trained_model
    else:
        return xgb.XGBClassifier()


def get_search(xgb_model: XGBClassifier):
    c = Configuration().get()
    rsc = c["RANDOMIZED_SEARCH_CONFIG"]

    random_search = RandomizedSearchCV(
        estimator=xgb_model,
        param_distributions=rsc["param_distributions"],
        n_iter=rsc["n_iter"],
        scoring=rsc["scoring"],
        error_score=rsc["error_score"],
        cv=rsc["cv"],
        verbose=rsc["verbose"],
        random_state=rsc["random_state"],
    )
    return random_search


def get_best_parameters(random_search: RandomizedSearchCV) -> dict[Any, Any]:
    return random_search.best_params_


def get_best_score(random_search: RandomizedSearchCV) -> dict[Any, Any]:
    return random_search.best_score_


def get_best_model(random_search: RandomizedSearchCV) -> XGBClassifier:
    return random_search.best_estimator_


@dataclass
class TuningOutput:
    best_model: XGBClassifier
    best_params: any
    best_score: any


def tune(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    pre_trained_model: XGBClassifier | None = None,
    sample_weights: pd.DataFrame | None = None,
):
    # Define the XGBoost model and RandomSearch
    xgb_model = get_classifier(pre_trained_model=pre_trained_model)
    random_search = get_search(xgb_model)

    # Fit the model to the training data with or without weights
    random_search.fit(X_train, y_train, sample_weights=sample_weights)

    # Get the best parameters, best score and best model
    return TuningOutput(
        best_model=get_best_model(random_search),
        best_params=get_best_parameters(random_search),
        best_score=get_best_score(random_search),
    )

@dataclass
class ModelEvaluation:
    tuning_output: TuningOutput
    auc_score: float
    f1_score: float

def get_model_evaluation(
    tuning_output: TuningOutput, X_test: pd.DataFrame, y_test: pd.DataFrame
) -> ModelEvaluation:
    model = tuning_output.best_model 
    y_pred = model.predict(X_test)
    auc = roc_auc_score(y_test, y_p
                        red)
    f1 = f1_score(y_test, y_pred)
    return ModelEvaluation(tuning_output=tuning_output,
                           auc_score=auc, 
                           f1_score=f1)
