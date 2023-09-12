from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from itertools import combinations_with_replacement
from typing import Any, List, Tuple

import numpy as np


@dataclass
class RndParameter(ABC):
    """
    Abstract base class for random paramaters of a data-generating process.
    """

    @abstractmethod
    def generate(self) -> Any:
        ...


@dataclass
class BaseVariableNames:
    """
    Generates Names of Base Variables.
    Will be used as features names in the generated datasets

    Arg:
        size: number of variables
    """

    size: int
    base_var_names: List[str] = field(default_factory=list, init=False)

    def generate(self):
        return [f"X{i}" for i in range(self.size)]

    def __post_init__(self):
        self.base_var_names = self.generate()


@dataclass
class CensoredVariables:
    """
    A subset of the base variables
    """

    censored__variables: List[str]


@dataclass
class RndCensoredVariables(RndParameter):
    """
    Randomly picks which variables to censor

    Args:
        censored_size: number of base variables that should be censored
        total_size: number of base variables to pick from

    """

    seed: int
    censored_size: int
    total_size: int
    censored__variables: List[str] = field(init=False, default_factory=list)

    def generate(self) -> List[str]:
        available_vars = BaseVariableNames(self.total_size).base_var_names
        np.random.seed(self.seed + 2365)
        censored = np.random.choice(
            available_vars, size=int(self.censored_size), replace=False
        ).tolist()
        return censored

    def __post_init__(self):
        self.censored__variables = self.generate()


@dataclass
class InteractionVariableNames:
    """
    Generates names of Interaction Variables
    """

    size: int
    intr_separ_names: List[Tuple[str, str]] = field(default_factory=list, init=False)
    intr_joint_name: List[str] = field(default_factory=list, init=False)

    def generate_name_pairs(self):
        base_var_names = BaseVariableNames(self.size).base_var_names
        pairs = list(combinations_with_replacement(base_var_names, r=2))
        return pairs

    def generate_var_names(self, pairs: List[Tuple[str, str]]):
        return ["".join(c) for c in pairs]

    def __post_init__(self):
        self.intr_separ_names = self.generate_name_pairs()
        self.intr_joint_name = self.generate_var_names(pairs=self.intr_separ_names)


@dataclass
class Coefficients:
    """
    Coefficients of the "reverse logistic regression" DGP
    """

    coefficients: List[float]


@dataclass
class RndCoefficients(RndParameter):
    """
    Generates Random Coefficients of the "reverse logistic regression" DGP

    Args:
    size: number of coefficients
    prob_of_zero: probability of coefficient being zero
    same_sign: True if all coefficients should be positive, False if negative is allowed
    value_range: maximum absolut value of coefficients
    """

    size: int
    seed: int
    prob_of_zero: float
    same_sign: bool
    value_range: float = field(default=10)
    coefficients: List[float] = field(init=False, repr=True, default_factory=list)

    def generate(
        self,
    ) -> List[float]:
        np.random.seed(self.seed + 568)
        upper_bound = self.value_range
        lower_bound = 0 if self.same_sign else -self.value_range
        zeros = np.random.choice(
            [0, 1], p=[self.prob_of_zero, 1 - self.prob_of_zero], size=self.size
        )
        coefficients = np.random.randint(lower_bound, upper_bound, size=self.size)
        coefficients = coefficients * zeros
        return coefficients.tolist()

    def __post_init__(self):
        self.coefficients = self.generate()


@dataclass
class CorrellationMatrix:
    """
    Correlation Matrix of the DGP
    """

    _matrix: List[List[float]] = field(default_factory=list)
    matrix: Any = field(init=False)

    def __post_init__(self):
        self.matrix = np.array(self._matrix)


@dataclass
class RndCorrelationMatrix(RndParameter):
    """
    Generates a random Correlation Matrix of the DGP

    Args:
        size: dimension of the matrix
        same_sign: True if only positive correlations, False if negative correlations are allowed

    """

    size: int
    seed: int
    same_sign: bool
    value_range: float
    matrix: Any = field(init=False, repr=True)

    def generate(self) -> Any:
        np.random.seed(self.seed + 546783)

        if self.same_sign:
            matrix = np.random.uniform(0, self.value_range, size=(self.size, self.size))
        else:
            matrix = np.random.uniform(
                -self.value_range, self.value_range, size=(self.size, self.size)
            )
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
        self.matrix = self.generate()


@dataclass
class BernoulliBias:
    """
    Bias for the DGP
    """

    bias: float


@dataclass
class RndBernoulliBias(RndParameter):
    """
    Generates a random bias for the DGP
    """

    seed: int
    value_range: float
    bias: float = field(init=False)

    def generate(self) -> float:
        np.random.seed(self.seed + 456789)
        bias = np.random.uniform(-self.value_range, self.value_range)
        return bias

    def __post_init__(self):
        self.bias = self.generate()


@dataclass
class TransformationExponent:
    """Exponent for transforming features during the DGP"""

    exponent: float


@dataclass
class RndTransformationExponent(RndParameter):
    """Generates a random Transformation Exponent"""

    seed: int
    exponent_range_lower: float
    exponent_range_upper: float
    exponent: float = field(init=False)

    def generate(self) -> float:
        np.random.seed(self.seed + 6790)
        exponent = np.random.uniform(
            self.exponent_range_lower, self.exponent_range_upper
        )
        return exponent

    def __post_init__(self):
        self.exponent = self.generate()
