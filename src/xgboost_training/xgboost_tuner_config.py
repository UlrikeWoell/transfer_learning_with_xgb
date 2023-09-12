from dataclasses import dataclass
from typing import List


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