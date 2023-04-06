from config import GRIDSEARCH_CONFIG, RANDOMIZED_SEARCH_CONFIG
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


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
            )
        else:
            raise KeyError("search_type must be ranodm or grid")

    def get(self):
        return self.search
