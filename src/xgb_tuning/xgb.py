from abc import ABC, abstractmethod

import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from xgboost import XGBClassifier

from src.util.config_reader import Configuration
from src.util.timer import Timer

c = Configuration().get()
XGB_STANDARD_CONFIG = c['XGB_STANDARD_CONFIG']
GRIDSEARCH_CONFIG = c['GRIDSEARCH_CONFIG']
RANDOMIZED_SEARCH_CONFIG = c['RANDOMIZED_SEARCH_CONFIG']

class SearchSetUp:
    def __init__(self, search_type: str, clf) -> None:

        if search_type == "grid":
            self.search = GridSearchCV(
                estimator=clf,
                param_grid=GRIDSEARCH_CONFIG["param_grid"],
                cv=GRIDSEARCH_CONFIG["cv"],
            )
        elif search_type == "random":
            self.search = RandomizedSearchCV(
                estimator=clf,
                param_distributions=RANDOMIZED_SEARCH_CONFIG["param_distributions"],
                cv=RANDOMIZED_SEARCH_CONFIG["cv"],
                random_state=RANDOMIZED_SEARCH_CONFIG["random_state"],
                n_iter=RANDOMIZED_SEARCH_CONFIG["n_iter"],
                scoring="neg_mean_squared_error",
                error_score=RANDOMIZED_SEARCH_CONFIG['error_score']
            )
        else:
            raise KeyError("search_type must be ranodm or grid")

    def get(self):
        return self.search

class XGBTuner(ABC):
    def __init__(self) -> None:
        self.timer: Timer | None = Timer()
        self.xgb = XGBClassifier()
        self.search : SearchSetUp
        

    def tune(self, x_features:pd.DataFrame, y_target: pd.Series):
        print('Start Tuning')
        self.timer.start()
        # acts like gridsearch.fit(X, Y) or randomsearch.fit(X, Y)
        self.search.fit(X = x_features, y = y_target)
        self.timer.end()
        self.timer.print()
        print('Tuning finished')

class XGBTunerRand(XGBTuner):
    def __init__(self) -> None:
        super().__init__()
        self.search = SearchSetUp(search_type="random", clf=self.xgb).get()


class XGBTunerGrid(XGBTuner):
    def __init__(self) -> None:
        super().__init__()
        self.search = SearchSetUp(search_type="grid", clf=self.xgb).get()

