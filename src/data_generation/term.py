import itertools as it
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import partial
from typing import Generator

from src.data_generation.abstractfactory import AbstractFactory
from src.data_generation.distribution import AbstractDistribution


@dataclass
class Term(ABC):
    """
    Handles one covariate and its coefficient
    """

    coefficient: float
    variable_name: str
    variable_distribution: AbstractDistribution

    """
    def __str__(self) -> str:
        d_type = self.variable_distribution.distribution_type
        d_params = tuple(self.variable_distribution.distribution_params.values())
        return f"{self.coefficient}*{self.variable_name}  ~  {d_type} {d_params}"
    """


@dataclass
class TermFactory(AbstractFactory):
    variable_name_prefix: str
    names: Generator = field(default_factory=list, init=False)

    def __post_init__(self):
        self.names = self._generate_names(self.variable_name_prefix)

    def _generate_names(self, prefix: str) -> Generator:
        return (f"{prefix.upper()}{i}" for i in it.count(start=1, step=1))

    def produce_one(
        self,
        coeff_dist_var_dist_tuple: tuple[AbstractDistribution,AbstractDistribution]
    ) -> Term:
        coeff_dist = coeff_dist_var_dist_tuple[0]
        var_dist = coeff_dist_var_dist_tuple[1]
        return Term(
            coefficient=coeff_dist.sample_n(1),
            variable_name=self._get_variable_name(),
            variable_distribution=var_dist,
        )

    def _get_variable_name(self):
        return next(self.names)
