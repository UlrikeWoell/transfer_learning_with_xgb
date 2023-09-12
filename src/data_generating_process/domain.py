from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import polars as pl
from more_itertools import first_true

from src.data_generating_process.domain_parameters import (
    BaseVariableNames,
    BernoulliBias,
    CensoredVariables,
    Coefficients,
    CorrellationMatrix,
    InteractionVariableNames,
    RndBernoulliBias,
    RndCensoredVariables,
    RndCoefficients,
    RndCorrelationMatrix,
    RndTransformationExponent,
    TransformationExponent,
)

# Defaults
NUMBER_OF_RND_CENSORED_FEATURES = 6
NUMBER_OF_BASE_FEATURES = 30
PROB_OF_ZERO_BASE_COEFF = 0.8
VALUE_RANGE_BASE_COEFF = 10
PROB_OF_ZERO_INTERACTION_COEFF = 0.5
VALUE_RANGE_INTERACTION_COEFF = 10
COEFF_SAME_SIGN = True
CORRELATION_VALUE_RANGE = 0.5
CORRELATION_SAME_SIGN = False
BIAS_VALUE_RANGE = 0.3
EXPONENT_LOWER = 0.5
EXPONENT_UPPPER = 2
PROB_OF_CENSORING = 0.2


@dataclass
class BaseTerm:
    name: str
    coef: float


@dataclass
class IntrTerm:
    name1: str
    name2: str
    coef: float


@dataclass
class Domain:
    base_coeff_cnt: int
    base_variable_names: BaseVariableNames
    intr_variable_names: InteractionVariableNames
    correllation_matrix: RndCorrelationMatrix | CorrellationMatrix
    base_coefficients: RndCoefficients | Coefficients
    intr_coefficients: RndCoefficients | Coefficients
    bias: RndBernoulliBias | BernoulliBias
    transformation_exponent: RndTransformationExponent | TransformationExponent
    censored_variables: RndCensoredVariables | CensoredVariables

    @property
    def base_terms(self) -> List[BaseTerm]:
        base_vars = [
            BaseTerm(name=name, coef=coef)
            for name, coef in zip(
                self.base_variable_names.base_var_names,
                self.base_coefficients.coefficients,
            )
        ]
        return base_vars

    @property
    def intr_terms(self) -> List[IntrTerm]:
        intr_terms = [
            IntrTerm(name1=name1, name2=name2, coef=coef)
            for name1, name2, coef in zip(
                self.intr_variable_names.intr_separ_names[0],
                self.intr_variable_names.intr_separ_names[1],
                self.intr_coefficients.coefficients,
            )
        ]
        return intr_terms


@dataclass
class SamplingParameters:
    n: int
    seed: int


class DomainSampler:
    """
    Samples data from a given domain
    """

    @staticmethod
    def generate_X(domain: Domain, n: int, sample_seed: int) -> pl.DataFrame:
        """_summary_

        Args:
            domain (Domain): domain to sample from (provides number of features and feature names)
            n (int): number of samples to draw
            sample_seed (int): seed for reproducible sampling of data

        Returns:
            pl.DataFrame: Polars.DataFrame with multi-normal feature data (X)
        """
        rng = np.random.default_rng(seed=sample_seed)
        mean_vector = [0] * domain.base_coeff_cnt

        data = rng.multivariate_normal(
            mean=mean_vector,
            cov=domain.correllation_matrix.matrix,
            size=n,
            check_valid="warn",
            tol=0.01,
        )
        feature_names = domain.base_variable_names.base_var_names
        pl_df = pl.from_numpy(data, schema=feature_names, orient="row")

        return pl_df

    @staticmethod
    def transform_data(
        domain: Domain,
        X: pl.DataFrame,
    ) -> pl.DataFrame:
        """
        Transforms X with the transformation exponent obtained from the domain

        Args:
            domain (Domain): domain to sample from
            X (pl.DataFrame): untransformed feature data

        Returns:
            pl.DataFrame: Polars.DataFrame with transformed feature data
        """
        # raise feature data to the given power
        feature_names = domain.base_variable_names.base_var_names
        exponent = domain.transformation_exponent.exponent
        transformation_tasks = [
            (pl.col(f).sign() * pl.col(f).abs().pow(exponent)).alias(f)
            for f in feature_names
        ]
        X_powered = X.select(transformation_tasks)

        # re-scale the to have variance 1
        scales = X_powered.std()
        scale_tasks = [pl.col(n) / scales[n][0] for n in X_powered.columns]
        X_rescaled = X_powered.select(scale_tasks)

        return X_rescaled

    @staticmethod
    def aggregate(domain: Domain, X: pl.DataFrame) -> pl.DataFrame:
        """Aggregation function of the reverse logistic regression:
            Multiply each value of X and the interactions
            with the respective coefficient and sum them up.

        Args:
            domain (Domain): domain to sample from
            X (pl.DataFrame): feature data
        Returns:
            pl.DataFrame: with one newly added column containing
            the aggregated value
        """
        # prep multiplication of feature values with their coefficients
        base_multiplications = [
            (f.coef * pl.col(f.name)).alias(f.name) for f in domain.base_terms
        ]

        # prep multiplications of interactions (feat1 * feat2) with the coefficient
        interaction_multiplications = [
            (f.coef * pl.col(f.name1) * pl.col(f.name2)).alias(f"{f.name1}{f.name2}")
            for f in domain.intr_terms
        ]

        all_multiplications = base_multiplications + interaction_multiplications

        multiply_sum_across = (
            X.select(all_multiplications)  # do all the multiplication
            .fold(lambda s1, s2: s1 + s2)  # sum across
            .alias("agg")  # rename the Series
        )

        return pl.DataFrame(multiply_sum_across)

    @staticmethod
    def sigmoid_link(X: pl.DataFrame) -> pl.DataFrame:
        """
        Adds the class probability to X.

        Args:
            X (pl.DataFrame): feature data
        Returns:
            pl.DataFrame: with one newly added column ('link')
            containing the class probability

        """

        link = X.with_columns(
            (
                (pl.col("agg").exp() / (pl.col("agg").exp() + 1)).round(decimals=3)
                # e^x / (e^x + 1)
            ).alias("link")
        )
        return link

    @staticmethod
    def bernoulli_label(
        domain: Domain, X: pl.DataFrame, sample_seed: int
    ) -> pl.DataFrame:
        """Uses Bernoulli sampling to add a class label to the features

        Args:
            domain (Domain): domain to sample from, provides the Bias
            X (pl.DataFrame): feature data with probabilities
            sample_seed (int): for reproducibility

        Returns:
            pl.DataFrame: with one newly added column ('class')
            containing the class label
        """

        def bernoulli(X: pl.DataFrame, sample_seed: int) -> pl.DataFrame:
            """Helper function for sampling"""
            rng = np.random.default_rng(seed=sample_seed)

            probs = X.select(pl.col("biased_prob")).to_numpy().flatten()
            picks = (rng.uniform(size=None) < probs) * 1
            picks = pl.from_numpy(picks, schema=["class"])
            new = pl.concat([X, picks], how="horizontal")
            return new

        b = domain.bias.bias
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

    @staticmethod
    def generate_y(domain: Domain, X: pl.DataFrame, sample_seed: int) -> pl.DataFrame:
        """
        Reverse logistic regression to add class labels

        Args:
            domain (Domain): domain to sample from
            X (pl.DataFrame): feature matrix
            sample_seed (int): for reproducibility

        Returns:
            pl.DataFrame: with newly added columns for aggregtion, link, probability and class
        """
        X_agg = DomainSampler.aggregate(domain=domain, X=X)
        X_link = DomainSampler.sigmoid_link(X=X_agg)
        X_class = DomainSampler.bernoulli_label(
            domain=domain, X=X_link, sample_seed=sample_seed
        )
        return X_class

    @staticmethod
    def generate_data(domain: Domain, sample: SamplingParameters) -> pd.DataFrame:
        """
        Generates a dataset consisting of features (X) and labels (y).
        The distributional properties of the data are determined by the Domain.
        The size of the data is determined by the SamplingParameters

        Args:
            domain (Domain): domain to sample from
            sample (SamplingParameters): provides sample size and sample seed
        """
        n = sample.n
        sample_seed = sample.seed

        X = DomainSampler.generate_X(domain=domain, n=n, sample_seed=sample_seed)
        X = DomainSampler.transform_data(domain=domain, X=X)
        y = DomainSampler.generate_y(domain=domain, X=X, sample_seed=sample_seed)
        X = DomainSampler.censor_data(domain=domain, X=X)

        result = X.to_pandas()
        result["y"] = y.select(pl.col("class")).to_pandas()
        return result

    @staticmethod
    def censor_data(domain: Domain, X: pl.DataFrame) -> pl.DataFrame:
        n = X.shape[0]
        zeroes = [
            pl.zeros(n=n).alias(v)
            for v in domain.censored_variables.censored__variables
        ]
        censored_X = X.with_columns(zeroes)
        return censored_X


@dataclass
class DomainParameters:
    """Specifies the paramaters of a Domain.
    DomainGenerator class uses the DomainParameters class
    to generate the described domain.

    If parameters are provided, they will be used.
    If they are None, they will be generated randomly by the DomainGenerator
    """

    base_coeff_cnt: int
    base_variable_names: BaseVariableNames | None = field(init=False)
    intr_variable_names: InteractionVariableNames | None = field(init=False)
    correllation_matrix: CorrellationMatrix | RndCorrelationMatrix | None = field(
        default=None
    )
    base_coefficients: Coefficients | RndCoefficients | None = field(default=None)
    base_coefficients_prob_of_zero: float | None = field(default=None)
    intr_coefficients: Coefficients | RndCoefficients | None = field(default=None)
    intr_coefficients_prob_of_zero: float | None = field(default=None)
    coeff_same_sign: bool | None = field(default=None)
    bias: BernoulliBias | RndBernoulliBias | None = field(default=None)
    transformation_exponent: TransformationExponent | RndTransformationExponent | None = field(
        default=None
    )
    censored_variables: CensoredVariables | RndCensoredVariables | None = field(
        default=None
    )
    number_of_censored_variables: int | None = field(default=None)

    def __post_init__(self):
        self.base_variable_names = BaseVariableNames(self.base_coeff_cnt)
        self.intr_variable_names = InteractionVariableNames(self.base_coeff_cnt)


class DomainGenerator:
    """
    Generate a Domain using the provided parameters. 
    Replaces missing parameters with random parameters using reasonable defaults

    Returns:
        _type_: _description_
    """
    
    @staticmethod
    def get_base_coeff_cnt(params: DomainParameters):
        return first_true(
            [params.base_coeff_cnt, NUMBER_OF_BASE_FEATURES],
            pred=lambda x: x is not None,
        )

    @staticmethod
    def get_base_variable_names(params: DomainParameters):
        return params.base_variable_names

    @staticmethod
    def get_intr_variable_names(params: DomainParameters):
        return params.intr_variable_names

    @staticmethod
    def get_correllation_matrix(params: DomainParameters, seed: int):
        return first_true(
            [
                params.correllation_matrix,
                RndCorrelationMatrix(
                    params.base_coeff_cnt,
                    seed=seed,
                    same_sign=CORRELATION_SAME_SIGN,
                    value_range=CORRELATION_VALUE_RANGE,
                ),
            ],
            pred=lambda x: x is not None,
        )

    @staticmethod
    def get_base_coefficients(params: DomainParameters, seed: int):
        base_coefficients_prob_of_zero = first_true(
            [params.base_coefficients_prob_of_zero, PROB_OF_ZERO_BASE_COEFF],
            pred=lambda x: x is not None,
        )

        same_sign = first_true(
            [params.coeff_same_sign, COEFF_SAME_SIGN],
            pred=lambda x: x is not None,
        )

        return first_true(
            [
                params.base_coefficients,
                RndCoefficients(
                    size=params.base_coeff_cnt,
                    seed=seed,
                    prob_of_zero=base_coefficients_prob_of_zero,
                    same_sign=same_sign,
                ),
            ],
            pred=lambda x: x is not None,
        )

    @staticmethod
    def get_intr_coefficients(params: DomainParameters, seed: int):
        intr_coefficients_prob_of_zero = first_true(
            [params.intr_coefficients_prob_of_zero, PROB_OF_ZERO_INTERACTION_COEFF],
            pred=lambda x: x is not None,
        )

        same_sign = first_true(
            [params.coeff_same_sign, COEFF_SAME_SIGN],
            pred=lambda x: x is not None,
        )

        return first_true(
            [
                params.intr_coefficients,
                RndCoefficients(
                    size=len(params.intr_variable_names.intr_joint_name),
                    seed=seed,
                    prob_of_zero=intr_coefficients_prob_of_zero,
                    same_sign=same_sign,
                ),
            ],
            pred=lambda x: x is not None,
        )

    @staticmethod
    def get_censored_variables(params: DomainParameters, seed: int):
        number_of_censored_variables = first_true(
            [
                params.number_of_censored_variables,
                round(PROB_OF_CENSORING * params.base_coeff_cnt),
            ],
            pred=lambda x: x is not None,
        )
        return first_true(
            [
                params.censored_variables,
                RndCensoredVariables(
                    seed=seed,
                    censored_size=number_of_censored_variables,
                    total_size=params.base_coeff_cnt,
                ),
            ],
            pred=lambda x: x is not None,
        )

    @staticmethod
    def get_bias(params: DomainParameters, seed: int):
        return first_true(
            [params.bias, RndBernoulliBias(seed=seed, value_range=BIAS_VALUE_RANGE)],
            pred=lambda x: x is not None,
        )

    @staticmethod
    def get_transformation_exponent(params: DomainParameters, seed: int):
        return first_true(
            [
                params.transformation_exponent,
                RndTransformationExponent(
                    seed=seed,
                    exponent_range_lower=EXPONENT_LOWER,
                    exponent_range_upper=EXPONENT_UPPPER,
                ),
            ],
            pred=lambda x: x is not None,
        )

    @classmethod
    def get_domain(cls, params: DomainParameters, seed: int):
        """
        Creates a domain accorind to the specified params. If domain parameters are not
        provided, they will be replaced by a random parameter, generated using seed.

        Args:
            params (DomainParameters): povides fixed parameters for the domain
            seed (int): seed for creating random missing parameters that are not part of params

        Returns:
            Domain: domain with parameters are specified or random
        """
        base_coeff_cnt = cls.get_base_coeff_cnt(params)
        base_variable_names = cls.get_base_variable_names(params)
        intr_variable_names = cls.get_intr_variable_names(params)
        correllation_matrix = cls.get_correllation_matrix(params, seed)
        base_coefficients = cls.get_base_coefficients(params, seed)
        intr_coefficients = cls.get_intr_coefficients(params, seed)
        bias = cls.get_bias(params, seed)
        transformation_exponent = cls.get_transformation_exponent(params, seed)
        censored_variables = cls.get_censored_variables(params, seed)

        return Domain(
            base_coeff_cnt=base_coeff_cnt,
            base_variable_names=base_variable_names,
            intr_variable_names=intr_variable_names,
            correllation_matrix=correllation_matrix,
            base_coefficients=base_coefficients,
            intr_coefficients=intr_coefficients,
            bias=bias,
            transformation_exponent=transformation_exponent,
            censored_variables=censored_variables,
        )
