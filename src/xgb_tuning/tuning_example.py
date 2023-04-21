import warnings

import pandas as pd
from sklearn import datasets

from src.xgb_tuning.xgb import XGBTunerGrid, XGBTunerRand

warnings.filterwarnings("ignore")


iris = datasets.load_iris()
X = pd.DataFrame(iris.data[:, :4])
X.columns = iris["feature_names"]
Y = iris.target

tuner = XGBTunerRand(X, Y)
tuner.tune()
