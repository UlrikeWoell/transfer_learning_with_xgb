from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from itertools import combinations_with_replacement
from pprint import pprint
from random import choices, randint, random, sample, seed, uniform
from typing import Any, Callable, List, Tuple

import numpy as np
import pandas as pd
import polars as pl


@dataclass
class Term:
    coefficient: float
    var_name: str


@dataclass
class InteractionTerm:
    coefficient: float
    var_name_1: str
    var_name_2: str


@dataclass
class Domain:
    domain_seed: int
    base_feature_count: int
    base_features: List[Term] = field(default_factory=list, init=False)
    censored_features: List[Term] = field(default_factory=list, init=False)
    feature_covariance: np.ndarray[Any, Any] = field(init=False)
    interactions: List[InteractionTerm] = field(default_factory=list, init=False)
    noise_variance: float = field(init=False)
    bernoulli_bias: float = field(init=False)
    use_copula: bool = field(init=False)
    power_transformation: float = field(init=False)

    def __post_init__(self):
        # make reproducible
        seed(self.domain_seed)

        # build domain
        self.base_features = self.create_base_features(self.base_feature_count)
        self.feature_covariance = self.create_feature_covariance(
            self.base_feature_count
        )
        self.interactions = self.create_interactions(self.base_features)
        self.noise_variance = self.get_noise_variance(self.base_feature_count)
        self.bernoulli_bias = self.get_bernoulli_bias()
        self.censored_features = []
        self.use_copula = False
        self.power_transformation = 1

    def aggregate(self, X: pl.DataFrame) -> pl.DataFrame:
        # prep multiplication of feature values with their coefficients
        feature_multiplications = [
            (f.coefficient * pl.col(f.var_name)).alias(f.var_name)
            for f in self.base_features
        ]

        # prep multiplications of interactions (feat1 * feat2) with the coefficient
        interaction_multiplications = [
            (f.coefficient * pl.col(f.var_name_1) * pl.col(f.var_name_2)).alias(
                f"{f.var_name_1}{f.var_name_2}"
            )
            for f in self.interactions
        ]

        all_multiplications = feature_multiplications + interaction_multiplications

        multiply_sum_across = (
            X.select(all_multiplications)
            .fold(lambda s1, s2: s1 + s2)  # do all the multiplication
            .alias("agg")  # sum across  # rename the Series
        )

        return pl.DataFrame(multiply_sum_across)

    def sigmoid_link(self, X: pl.DataFrame) -> pl.DataFrame:
        """Maps a number to the (0,1) interval"""

        link = X.with_columns(
            (
                (pl.col("agg").exp() / (pl.col("agg").exp() + 1)).round(decimals=3)
                # e^x / (e^x + 1)
            ).alias(  # logistic function
                "link"
            )
        )  # rename
        return link

    def bernoulli_sampler(self, X: pl.DataFrame, sample_seed: int) -> pl.DataFrame:
        "Returns a bernoulli_sampler function with bias"

        def bernoulli(X: pl.DataFrame, sample_seed: int) -> pl.DataFrame:
            rng = np.random.default_rng(seed=sample_seed)

            probs = X.select(pl.col("biased_prob")).to_numpy().flatten()
            picks = (rng.uniform(size=None) < probs) * 1
            picks = pl.from_numpy(picks, schema=["class"])
            new = pl.concat([X, picks], how="horizontal")
            return new

        b = self.bernoulli_bias
        X_class = X.with_columns(
            [
                # add bias to probability but stay within [0,1]
                pl.when(pl.col("link") + b > 1)
                .then(1)
                .when(pl.col("link") + b < 0)
                .then(0)
                .otherwise(pl.col("link") + b)
                .alias("biased_prob"),
            ]
        )
        X_class = bernoulli(X=X_class, sample_seed=sample_seed)

        return X_class

    def generate_y(self, X: pl.DataFrame, sample_seed: int) -> pl.DataFrame:
        X_agg = self.aggregate(X)
        X_link = self.sigmoid_link(X_agg)
        X_class = self.bernoulli_sampler(X_link, sample_seed=sample_seed)
        return X_class

    def create_base_features(self, base_feature_count: int) -> List[Term]:
        """_summary_

        Args:
            feature_count (int): How many features should be created

        Returns:
            List[Term]: Features tuple(coeff, name)
        """
        names = [f"X{i}" for i in range(base_feature_count)]
        coeff_pool = list(range(-10, 10, 1))
        coeff_pool.remove(0)
        coeffs = list(choices(coeff_pool, k=base_feature_count))
        return [Term(t[0], t[1]) for t in zip(coeffs, names)]

    def create_feature_covariance(
        self, base_feature_count: int
    ) -> np.ndarray[Any, Any]:
        """Randomly creates a covariance matrix. Single features have variance = 1"""
        matrix = [[0] * base_feature_count for _ in range(base_feature_count)]

        for i in range(base_feature_count):
            for j in range(i + 1):
                if i == j:
                    matrix[i][j] = 1
                else:
                    matrix[i][j] = round(uniform(-0.7, 0.7), 3)
                    matrix[j][i] = matrix[i][j]

        return np.array(matrix)

    def create_interactions(self, base_features: List[Term]) -> List[InteractionTerm]:
        """
        Randomly creates interaction terms

        Args:
            base_features (int): features to pick from to create interactions

        Returns:
            List[Term]: Features tuple(coeff, name
        """
        feature_names = [t.var_name for t in base_features]
        names_pool = list(combinations_with_replacement(feature_names, r=2))
        coeff_pool = list(range(-10, 10, 1))
        coeff_pool.remove(0)
        number_of_interactions = len(base_features) % 10

        interaction_names = list(choices(names_pool, k=number_of_interactions))
        interaction_coefs = list(choices(coeff_pool, k=number_of_interactions))

        interactions = []
        for i in range(number_of_interactions):
            coeff = interaction_coefs[i]
            name1 = interaction_names[i][0]
            name2 = interaction_names[i][1]
            new_interaction = InteractionTerm(
                coefficient=coeff, var_name_1=name1, var_name_2=name2
            )
            interactions.append(new_interaction)
        return interactions

    def get_noise_variance(self, feature_count: int):
        """Randomly selects a variance for the noise term

        Args:
            feature_count (int): _description_

        Returns:
            _type_: _description_
        """
        return randint(1, feature_count)

    def get_bernoulli_bias(self):
        """Randomly selects a bias"""
        return (0.1 * random()).__round__(3)

    def generate_data(
        self, n: int, sample_seed: int
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        seed(sample_seed)
        X = self.generate_X(n=n, sample_seed=sample_seed)
        X = self.transform_data(X)
        # if self.use_copula:
        #    X = self.get_copula_data(X, n=n, sample_seed=sample_seed)

        y = self.generate_y(X=X, sample_seed=sample_seed)
        X = self.censor_data(X)

        return X.to_pandas(), y.select(pl.col("class")).to_pandas()

    def generate_X(self, n: int, sample_seed: int) -> pl.DataFrame:
        rng = np.random.default_rng(seed=sample_seed)
        mean_vector = [0] * self.base_feature_count

        data = rng.multivariate_normal(
            mean=mean_vector, cov=self.feature_covariance, size=n
        )
        feature_names = [feat.var_name for feat in self.base_features]
        df = pl.from_numpy(data, schema=feature_names, orient="row")

        return df

    def transform_data(
        self,
        X: pl.DataFrame,
    ) -> pl.DataFrame:
        # raise feature data to the give power
        transformation_tasks = [
            (
                pl.col(f.var_name).sign()
                * pl.col(f.var_name).abs().pow(self.power_transformation)
            ).alias(f.var_name)
            for f in self.base_features
        ]
        X_powered = X.select(transformation_tasks)

        # re-scale the to have variance 1
        scales = X_powered.std()
        scale_tasks = [pl.col(n) / scales[n][0] for n in X_powered.columns]
        X_rescaled = X_powered.select(scale_tasks)

        return X_rescaled

    def censor_data(self, X: pl.DataFrame):
        n = X.shape[0]
        zeroes = [
            pl.zeros(n=n).alias(feature.var_name) for feature in self.censored_features
        ]
        X = X.with_columns(zeroes)
        return X


class DomainChanger:
    @staticmethod
    def change_covariance_matrix(domain: Domain, factor: float) -> Domain:
        """For non-inductive transfer problems.
        Current covariance matrix will be multiplied by the factor to increase or decrease all individual correlations

        Args:
            factor must be between 0 and 1.4 to avoid ensure Var(X) = 1
        """

        if factor < 0 or factor > 1.4:
            raise ValueError("factor must be in interval [0, 1.4]")

        new_matrix = [
            [0] * domain.base_feature_count for _ in range(domain.base_feature_count)
        ]

        for i in range(domain.base_feature_count):
            for j in range(i + 1):
                if i == j:
                    new_matrix[i][j] = 1
                else:
                    old = domain.feature_covariance[i][j]
                    new = round(old * factor, 3)
                    new_matrix[i][j] = new
                    new_matrix[j][i] = new
        new_matrix = np.array(new_matrix)
        domain.feature_covariance = new_matrix
        return domain

    @staticmethod
    def change_generate_X(domain: Domain) -> Domain:
        """non-inductive transfer problems"""
        raise NotImplementedError
        return domain

    @staticmethod
    def change_feature_coefficients(
        domain: Domain, factor_range_low: float, factor_range_up: float
    ) -> Domain:
        """Assign different but similar values to feature coefficients

        Args:
            factor_range_low (float): bigger than 0.5
            factor_range_up (float): lower than 2

        Raises:
            ValueError: _description_
            ValueError: _description_
        """
        if factor_range_low < 0.5 or factor_range_up > 2:
            raise ValueError("Factor range should be inside [0.5, 2.0]")
        if factor_range_low >= factor_range_up:
            raise ValueError("factor_range_low must be smaller than factor_range_up")

        # make reproducible
        seed(round(1000 * factor_range_low * factor_range_up))
        new_base_features = []
        for c in domain.base_features:
            old_coeff = c.coefficient
            old_var_name = c.var_name
            random_factor = uniform(factor_range_low, factor_range_up)
            new_coeff = round(old_coeff * random_factor, 1)
            new_base_features.append(Term(new_coeff, old_var_name))

        domain.base_features = new_base_features
        return domain

    @staticmethod
    def change_interaction_coefficients(
        domain: Domain, factor_range_low: float, factor_range_up: float
    ) -> Domain:
        """Assign different but similar values to interaction coefficients

        Args:
            factor_range_low (float): bigger than 0.5
            factor_range_up (float): lower than 2

        Raises:
            ValueError: _description_
            ValueError: _description_
        """
        if factor_range_low < 0.5 or factor_range_up > 2:
            raise ValueError("Factor range should be inside [0.5, 2.0]")
        if factor_range_low >= factor_range_up:
            raise ValueError("factor_range_low must be smaller than factor_range_up")

        # make reproducible
        seed(round(1000 * factor_range_low * factor_range_up))
        new_interactions = []
        for c in domain.interactions:
            old_coeff = c.coefficient
            old_var_name_1 = c.var_name_1
            old_var_name_2 = c.var_name_2
            random_factor = uniform(factor_range_low, factor_range_up)
            new_coeff = round(old_coeff * random_factor, 1)
            new_interactions.append(
                InteractionTerm(new_coeff, old_var_name_1, old_var_name_2)
            )

        domain.interactions = new_interactions
        return domain

    @staticmethod
    def change_noise_variance(domain: Domain, factor: float) -> Domain:
        """inductive transfer problems
        Multiplies current noise variance with a factor
        """
        if factor < 0 or factor > 2:
            raise ValueError("factor must be in interval [0, 2]")

        domain.noise_variance = domain.noise_variance * factor
        return domain

    @staticmethod
    def change_bernoulli_bias(domain: Domain, factor: float) -> Domain:
        """
        Inductive transfer problems
        Current covariance matrix will be multiplied by the factor to increase or decrease all individual correlations

        Args:
            factor must be between 0 and 1.4 to avoid ensure Var(X) = 1"""
        if factor < -1.8 or factor > 1.8:
            raise ValueError("factor must be in interval [-1.8, +1.8]")

        domain.bernoulli_bias = domain.bernoulli_bias * factor
        return domain

    @staticmethod
    def change_censored_features(domain: Domain, probability=float) -> Domain:
        """Randomly censores some of the feautures"""
        if probability < 0 or probability > 1:
            raise ValueError("probability must be in [0, 1]")

        # make it reprducible
        seed(round(1000 * probability))

        # draw the censored features
        n_to_censor = round(domain.base_feature_count * probability)
        domain.censored_features = sample(domain.base_features, k=n_to_censor)
        return domain

    @staticmethod
    def change_to_copula(domain: Domain) -> Domain:
        domain.use_copula = True
        return domain

    @staticmethod
    def change_transformation(domain: Domain, power: float) -> Domain:
        domain.power_transformation = power
        return domain


ooo = DomainChanger()

orig = Domain(domain_seed=1, base_feature_count=3)
chan = Domain(domain_seed=1, base_feature_count=3)
chan = ooo.change_transformation(domain=chan, power=1)

Xo, yo = orig.generate_data(n=300, sample_seed=354)
Xc, yc = chan.generate_data(n=300, sample_seed=354)
