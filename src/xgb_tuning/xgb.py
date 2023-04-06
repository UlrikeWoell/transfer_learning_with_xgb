
from config import (GRIDSEARCH_CONFIG,
                    XGB_STANDARD_GONFIG,
                    RANDOMIZED_SEARCH_CONFIG)
from search import SearchSetUp
from mytimer import Timer
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn import datasets

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


# A parameter grid for XGBoost
timer = Timer()

iris = datasets.load_iris()
X = pd.DataFrame(iris.data[:, :4])
X.columns = iris['feature_names']
Y = iris.target


xgb = XGBClassifier(XGB_STANDARD_GONFIG)
search = SearchSetUp(search_type='random', clf=xgb).get()

timer.start()
#gridsearch.fit(X, Y)
#randomsearch.fit(X, Y)
search.fit(X, Y)
timer.end()
timer.print()
