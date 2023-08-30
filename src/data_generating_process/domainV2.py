from dataclasses import dataclass, field
from pprint import pprint
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
    RndCorrellationMatrix,
    RndTransformationExponent,
    TransformationExponent,
)


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
    correllation_matrix: RndCorrellationMatrix | CorrellationMatrix
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
    @staticmethod
    def generate_X(domain: Domain, n: int, sample_seed: int) -> pl.DataFrame:
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

    @staticmethod
    def bernoulli_label(
        domain: Domain, X: pl.DataFrame, sample_seed: int
    ) -> pl.DataFrame:
        "Returns a bernoulli_sampler function with bias"

        def bernoulli(X: pl.DataFrame, sample_seed: int) -> pl.DataFrame:
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

    def generate_y(domain: Domain, X: pl.DataFrame, sample_seed: int) -> pl.DataFrame:
        X_agg = DomainSampler.aggregate(domain=domain, X=X)
        X_link = DomainSampler.sigmoid_link(X=X_agg)
        X_class = DomainSampler.bernoulli_label(
            domain=domain, X=X_link, sample_seed=sample_seed
        )
        return X_class

    @staticmethod
    def generate_data(domain: Domain, sample: SamplingParameters) -> pd.DataFrame:
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
    DomainGenerator class uses the DomainParameters class to generate the described domain.
    """

    base_coeff_cnt: int
    base_variable_names: BaseVariableNames = field(init=False)
    intr_variable_names: InteractionVariableNames = field(init=False)
    correllation_matrix: CorrellationMatrix | RndCorrellationMatrix | None = field(
        default=None
    )
    base_coefficients: Coefficients | RndCoefficients | None = field(default=None)
    base_coefficients_prob_of_zero: float | None = field(default=None)
    intr_coefficients: Coefficients | None | RndCoefficients = field(default=None)
    intr_coefficients_prob_of_zero: float | None = field(default=None)
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
    @staticmethod
    def get_base_values(params: DomainParameters):
        return (
            params.base_coeff_cnt,
            params.base_variable_names,
            params.intr_variable_names,
        )

    @staticmethod
    def get_correllation_matrix(params: DomainParameters, seed: int, same_sign: bool):
        return first_true(
            [
                params.correllation_matrix,
                RndCorrellationMatrix(
                    params.base_coeff_cnt, seed=seed, same_sign=same_sign
                ),
            ],
            pred=lambda x: x is not None,
        )

    @staticmethod
    def get_base_coefficients(params: DomainParameters, seed: int):
        base_coefficients_prob_of_zero = first_true(
            [params.base_coefficients_prob_of_zero, 0.8],
            pred=lambda x: x is not None,
        )
        return first_true(
            [
                params.base_coefficients,
                RndCoefficients(
                    size=params.base_coeff_cnt,
                    seed=seed,
                    prob_of_zero=base_coefficients_prob_of_zero,
                    same_sign=True
                ),
            ],
            pred=lambda x: x is not None,
        )

    @staticmethod
    def get_intr_coefficients(params: DomainParameters, seed: int):
        intr_coefficients_prob_of_zero = first_true(
            [params.intr_coefficients_prob_of_zero, 0.50],
            pred=lambda x: x is not None,
        )
        return first_true(
            [
                params.intr_coefficients,
                RndCoefficients(
                    size=len(params.intr_variable_names.intr_joint_name),
                    seed=seed,
                    prob_of_zero=intr_coefficients_prob_of_zero,
                    same_sign=True
                ),
            ],
            pred=lambda x: x is not None,
        )

    @staticmethod
    def get_censored_variables(params: DomainParameters, seed: int):
        number_of_censored_variables = first_true(
            [params.number_of_censored_variables, 0.2 * params.base_coeff_cnt],
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
            [params.bias, RndBernoulliBias(seed=seed)], pred=lambda x: x is not None
        )

    @staticmethod
    def get_transformation_exponent(params: DomainParameters, seed: int):
        return first_true(
            [params.transformation_exponent, RndTransformationExponent(seed=seed)],
            pred=lambda x: x is not None,
        )

    @classmethod
    def get_domain(cls, params: DomainParameters, seed: int, matrix_same_sign: bool):
        base_coeff_cnt, base_variable_names, intr_variable_names = cls.get_base_values(
            params
        )
        correllation_matrix = cls.get_correllation_matrix(
            params, seed, same_sign=matrix_same_sign
        )
        base_coefficients = cls.get_base_coefficients(params, seed)
        intr_coefficients = cls.get_intr_coefficients(params, seed)
        bias = cls.get_bias(params, seed)
        transformation_exponent = cls.get_transformation_exponent(params, seed)
        censored_variables = cls.get_censored_variables(params, seed)

        dom = Domain(
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
        return dom

    @classmethod
    def vary_bias(
        cls, base_coeff_cnt: int, values: List[float], default_seed: int
    ) -> Dict[float, Domain]:
        list_of_domains = {}
        for v in values:
            bias = BernoulliBias(v)
            dps = DomainParameters(base_coeff_cnt=base_coeff_cnt, bias=bias)
            dom = cls.get_domain(dps, default_seed)
            list_of_domains[v] = dom
        return list_of_domains

    @classmethod
    def vary_transformation_exponent(
        cls, base_coeff_cnt: int, values: List[float], default_seed: int
    ) -> Dict[float, Domain]:
        list_of_domains = {}
        for v in values:
            exponent = TransformationExponent(v)
            dps = DomainParameters(
                base_coeff_cnt=base_coeff_cnt, transformation_exponent=exponent
            )
            dom = cls.get_domain(dps, default_seed)
            list_of_domains[v] = dom
        return list_of_domains

    @classmethod
    def vary_correllation_matrix(
        cls, base_coeff_cnt: int, matrix_seeds: List[int], default_seed: int
    ) -> Dict[int, Domain]:
        list_of_domains = {}
        for seed in matrix_seeds:
            correllation_matrix = RndCorrellationMatrix(size=base_coeff_cnt, seed=seed)
            dps = DomainParameters(
                base_coeff_cnt=base_coeff_cnt, correllation_matrix=correllation_matrix
            )
            dom = cls.get_domain(dps, default_seed)
            list_of_domains[seed] = dom
        return list_of_domains

    @classmethod
    def vary_coefficients(
        cls, base_coeff_cnt: int, coeff_seeds: List[int], default_seed: int
    ) -> Dict[int, Domain]:
        # calc number of Interactions
        vars = InteractionVariableNames(base_coeff_cnt)
        intr_coeff_cnt = len(vars.intr_joint_name)

        list_of_domains = {}
        for seed in coeff_seeds:
            base_coeffs = RndCoefficients(
                size=base_coeff_cnt, seed=seed, prob_of_zero=0
            )
            intr_coeffs = RndCoefficients(
                size=intr_coeff_cnt, seed=seed + 1, prob_of_zero=0.9
            )
            dps = DomainParameters(
                base_coeff_cnt=base_coeff_cnt,
                base_coefficients=base_coeffs,
                intr_coefficients=intr_coeffs,
            )
            dom = cls.get_domain(dps, default_seed)
            list_of_domains[seed] = dom
        return list_of_domains
