from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
from numpy import ndarray
from numpy.random import PCG64, Generator

from src.data_generation.term import Term

SamplingFunction = Callable[[int], float | ndarray]
TargetFunction = Callable[[pd.DataFrame], pd.Series]
LinkFunction = Callable[[pd.Series], pd.Series]


@dataclass
class DataGenerationStrategy(ABC):
    """
    Defines a Data Generation Stratgy
    """

    terms: list[Term] = field(default_factory=list, repr=False)
    target_fct: TargetFunction | None = None
    link_fct: LinkFunction | None = None


class DataGenerator(ABC):
    def _generate_feature_data(self, terms: list[Term], n: int) -> pd.DataFrame:
        data = pd.DataFrame()
        for t in terms:
            data[f"{t.variable_name}"] = t.variable_distribution.sample_n(n=n)
        return data

    def _generate_target_data(
        self, target_fct: TargetFunction, feature_data: pd.DataFrame, terms: list[Term]
    ) -> pd.Series:
        target = target_fct(feature_data, terms)
        return target

    def _generate_transformed_target_data(
        self, link_fct: LinkFunction, target_data: pd.Series
    ) -> pd.Series:
        transformed_target = link_fct(target_data)
        return transformed_target

    def generate_data(
        self, strategy: DataGenerationStrategy, n: int
    ) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
        """
        Generates features, target and transformed target data according to the given strategy

        Args:
            strategy (DataGenerationStrategy): defines how to generate samples
            n (int): number of samples to create

        Returns:
            tuple[pd.DataFrame, pd.Series, pd.Series]: features, target, 0-1 transformed target
        """
        features = self._generate_feature_data(strategy.terms, n)
        target = self._generate_target_data(
            strategy.target_fct, features, strategy.terms
        )
        transformed_target = self._generate_transformed_target_data(
            strategy.link_fct, target
        )
        return features, target, transformed_target


def additive_target_fct(feature_data: pd.DataFrame, terms: list[Term]) -> pd.Series:
    y_raw = np.zeros(shape=feature_data.shape[0])
    for term in terms:
        var = term.variable_name
        y_raw = y_raw + term.coefficient * feature_data[var]
    return y_raw


def threshold_link_fct(
    untransformed_target: pd.Series, threshold: float = 0
) -> pd.Series:
    """
    Deterministic function to transform a variable into 0/1 class
    depending on fixed threshold
    Return class = 0 if Y is below or equal to threshold
    Retrun class = 1 if Y is above threshold

    Args:
        y: a pd.Series containing a variable that should be transformed
        into 0/1 class
        threshold: float that is used as cut-off value
    """
    y_class = np.where(untransformed_target <= threshold, 0, 1)
    return y_class


def logit_deterministic_link_fct(
    untransformed_target: pd.Series, probability_threshold: float = 0.5
) -> pd.Series:
    """
    Deterministic function to transform a variable into 0/1 class
    depending on probability threshold
    Return class = 0 if Y is below or equal to probability threshold
    Retrun class = 1 if Y is above threshold

    Args:
        y: a pd.Series containing a variable that should be transformed
        into 0/1 class
        threshold: float that is used as cut-off value
    """
    probabilities = 1 / (1 + np.exp(untransformed_target))
    y_class = threshold_link_fct(probabilities, probability_threshold)
    return y_class


def logit_random_link_fct(y: pd.Series, seed=1) -> pd.Series:
    """
    Random function to transform a variable into 0/1 class depending on
    probability.
    Probability is calculated by logit.
    Class is drawn from binomial distribution.

    Args:
        y: a pd.Series containing a variable that should be transformed into 0/1 class
        seed: seed for drawing from binomial
    """
    g = Generator(PCG64(seed=seed))
    probabilities = 1 / (1 + np.exp(y))
    y_class = [g.binomial(1, p, 1)[0] for p in probabilities]
    y_class = pd.Series(y_class)
    return y_class
