from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List

import pandas as pd
import xgboost as xgb
from sklearn.exceptions import NotFittedError
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from src.util.config_reader import Configuration


@dataclass
class TuningOutput:
    booster: xgb.Booster
    best_params: dict = field(default_factory=dict)


@dataclass
class ModelEvaluation:
    tuning_output: TuningOutput
    auc_score: float
    f1_score: float
    key: str | None = field(default=None)


le = LabelEncoder()


def get_classifier(pre_trained_model: XGBClassifier | None):
    if pre_trained_model:
        return pre_trained_model
    else:
        return xgb.XGBClassifier()


def get_search_search(
    xgb_model: XGBClassifier, pre_trained_search: RandomizedSearchCV | None
):
    c = Configuration().get()
    rsc = c["RANDOMIZED_SEARCH_CONFIG"]

    if pre_trained_search is None:
        return RandomizedSearchCV(
            estimator=xgb_model,
            param_distributions=rsc["param_distributions"],
            n_iter=rsc["n_iter"],
            scoring=rsc["scoring"],
            error_score=rsc["error_score"],
            cv=rsc["cv"],
            verbose=rsc["verbose"],
            random_state=rsc["random_state"],
        )
    else:
        return pre_trained_search


def get_search_custom_params(xgb_model: XGBClassifier, params: dict):
    c = Configuration().get()
    rsc = c["RANDOMIZED_SEARCH_CONFIG"]

    random_search = RandomizedSearchCV(
        estimator=xgb_model,
        param_distributions=params,
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


def simple_fit(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    model: XGBClassifier,
) -> TuningOutput:
    y_train = le.fit_transform(y_train["class"])
    model.fit(X_train, y_train)
    return TuningOutput(booster=model.get_booster(), best_params=model.get_params())


def continue_fit(
    X_train: pd.DataFrame, y_train: pd.DataFrame, pre_trained_model: xgb.Booster,
    params: dict
) -> TuningOutput:
    y_train = le.fit_transform(y_train["class"])
    dmatrix = xgb.DMatrix(X_train, y_train)
    full_model = xgb.train(dtrain=dmatrix, params = params, xgb_model=pre_trained_model)
    return TuningOutput(
        booster=full_model, best_params=xgb.XGBClassifier(full_model).get_params()
    )


def reweight_leaves(
    src_model: xgb.Booster,
    merged_X: pd.DataFrame,
    merged_y: pd.DataFrame,
):
    merged_y = le.fit_transform(merged_y["class"])

    # switch to sklearnAPI to get parameters
    n_rounds = xgb.XGBClassifier(src_model).get_params()["n_estimators"]

    Xy_refresh = xgb.DMatrix(merged_X.astype(float), merged_y.astype(int))
    refreshed = xgb.train(
        {"process_type": "update", "updater": "refresh", "refresh_leaf": True},
        Xy_refresh,
        num_boost_round=n_rounds,  # how many trees should be updated
        xgb_model=src_model,
    )

    return TuningOutput(
        booster=refreshed, best_params=xgb.XGBClassifier(booster=refreshed).get_params()
    )


def prune_trees(
    src_model: xgb.Booster,
    merged_X: pd.DataFrame,
    merged_y: pd.DataFrame,
):
    merged_y = le.fit_transform(merged_y["class"])
    Xy_refresh = xgb.DMatrix(merged_X, merged_y)
    refreshed = xgb.train(
        {"process_type": "update", "updater": "prune"},
        Xy_refresh,
        xgb_model=src_model,
    )

    return TuningOutput(
        booster=refreshed, best_params=xgb.XGBClassifier(booster=refreshed).get_params()
    )


def tune(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    pre_trained_model: XGBClassifier | None = None,
    pre_trained_search: RandomizedSearchCV | None = None,
    sample_weights: pd.DataFrame | None = None,
):
    # Define the XGBoost model and RandomSearch
    xgb_model = get_classifier(pre_trained_model=pre_trained_model)
    random_search = get_search_search(xgb_model, pre_trained_search)

    # Fit the model to the training data with or without weights
    y_train = le.fit_transform(y_train["class"])
    random_search.fit(
        X_train,
        y_train,  # weights=sample_weights
    )
    best_model = get_best_model(random_search)

    # Get the best parameters, best score and best model
    return TuningOutput(
        booster=best_model.get_booster(),
        best_params=get_best_parameters(random_search),
    )


def get_model_evaluation(
    tuning_output: TuningOutput,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    key: str | None = None,
) -> ModelEvaluation:
    model = tuning_output.booster

    dX = xgb.DMatrix(X_test)
    y_pred = model.predict(dX)

    auc = roc_auc_score(y_test, y_pred)
    # f1 = f1_score(y_test['class'], y_pred)
    return ModelEvaluation(
        tuning_output=tuning_output, auc_score=auc, f1_score="f1", key=key
    )
