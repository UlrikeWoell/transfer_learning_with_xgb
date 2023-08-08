from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from itertools import combinations_with_replacement
from pprint import pprint
from typing import Any, Callable, List, Tuple

import numpy as np
import pandas as pd
import polars as pl


@dataclass
class RndParameter(ABC):
    @classmethod
    @abstractmethod
    def generate(cls):
        ...


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
class BaseVariableNames:
    size: int
    base_var_names: List[str] = field(default_factory=list, init=False)

    @classmethod
    def generate(cls, size: int):
        return [f"X{i}" for i in range(size)]

    def __post_init__(self):
        self.base_var_names = BaseVariableNames.generate(self.size)


@dataclass
class CensoredVariables:
    censored__variables: List[str]


@dataclass
class RndCensoredVariables(RndParameter):
    seed: int
    censored_size: int
    total_size: int
    censored__variables: List[str] = field(init=False, default_factory=list)

    @classmethod
    def generate(cls, seed: int, censored_size: int, total_size: int) -> List[str]:
        available_vars = BaseVariableNames(total_size).base_var_names
        np.random.seed(seed + 2365)
        censored = np.random.choice(
            available_vars, size=int(censored_size), replace=False
        ).tolist()
        return censored

    def __post_init__(self):
        self.censored__variables = RndCensoredVariables.generate(
            seed=self.seed,
            censored_size=self.censored_size,
            total_size=self.total_size,
        )


@dataclass
class InteractionVariableNames:
    size: int
    intr_separ_names: List[Tuple[str, str]] = field(default_factory=list, init=False)
    intr_joint_name: List[str] = field(default_factory=list, init=False)

    @classmethod
    def generate_name_pairs(cls, size: int):
        base_var_names = BaseVariableNames.generate(size)
        pairs = list(combinations_with_replacement(base_var_names, r=2))
        return pairs

    @classmethod
    def generate_var_names(cls, pairs: List[Tuple[str, str]]):
        return ["".join(c) for c in pairs]

    def __post_init__(self):
        self.intr_separ_names = InteractionVariableNames.generate_name_pairs(self.size)
        self.intr_joint_name = InteractionVariableNames.generate_var_names(
            pairs=self.intr_separ_names
        )


@dataclass
class Coefficients:
    coefficients: List[float]


@dataclass
class RndCoefficients(RndParameter):
    size: int
    seed: int
    prob_of_zero: float
    lower_bound: float = field(default=-10)
    upper_bound: float = field(default=10)
    coefficients: List[float] = field(init=False, repr=True, default_factory=list)

    @classmethod
    def generate(
        cls,
        size: int,
        seed: int,
        lower_bound: int,
        upper_bound: int,
        prob_of_zero: float,
    ) -> List[float]:
        np.random.seed(seed + 568)
        zeros = np.random.choice([0, 1], p=[prob_of_zero, 1 - prob_of_zero], size=size)
        coefficients = np.random.randint(lower_bound, upper_bound, size=size)
        coefficients = coefficients * zeros
        return coefficients.tolist()

    def __post_init__(self):
        self.coefficients = RndCoefficients.generate(
            size=self.size,
            seed=self.seed,
            lower_bound=self.lower_bound,
            upper_bound=self.upper_bound,
            prob_of_zero=self.prob_of_zero,
        )


@dataclass
class CorrellationMatrix:
    _matrix: List[List[float]] = field(default_factory=list)
    matrix: np.ndarray = field(init=False)

    def __post_init__(self):
        self.matrix = np.array(self._matrix)


@dataclass
class RndCorrellationMatrix(RndParameter):
    size: int
    seed: int
    matrix: np.ndarray = field(init=False, repr=True)

    @classmethod
    def generate(cls, size: int, seed: int) -> np.ndarray:
        np.random.seed(seed + 546783)
        matrix = np.random.uniform(-1, 1, size=(size, size))
        matrix = (matrix + matrix.T) / 2  # Make the matrix symmetrical
        # Create a positive semidefinite matrix
        w, v = np.linalg.eigh(matrix)
        w = np.maximum(w, 0)
        matrix = np.dot(v, np.dot(np.diag(w), v.T))

        # Normalize the diagonal to 1 again, as numerical errors might have changed them slightly
        d = np.sqrt(np.diag(matrix))
        matrix /= d[:, None]
        matrix /= d[None, :]
        return matrix

    def __post_init__(self):
        self.matrix = RndCorrellationMatrix.generate(size=self.size, seed=self.seed)


@dataclass
class BernoulliBias:
    bias: float


@dataclass
class RndBernoulliBias(RndParameter):
    seed: int
    bias: float = field(init=False)

    @classmethod
    def generate(cls, seed: int) -> float:
        np.random.seed(seed + 456789)
        bias = np.random.uniform(-0.2, 0.2)
        return bias

    def __post_init__(self):
        self.bias = RndBernoulliBias.generate(self.seed)


@dataclass
class TransformationExponent:
    exponent: float


@dataclass
class RndTransformationExponent(RndParameter):
    seed: int
    exponent: float = field(init=False)

    @classmethod
    def generate(cls, seed: int) -> float:
        np.random.seed(seed + 6790)
        exponent = np.random.uniform(0.5, 2)
        return exponent

    def __post_init__(self):
        self.exponent = RndTransformationExponent.generate(self.seed)
