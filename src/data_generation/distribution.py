import itertools as it
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import partial
from typing import Callable, Literal,Type

import pandas as pd
from numpy import ndarray
from numpy.random import PCG64, Generator

from src.data_generation.abstractfactory import AbstractFactory
from src.util.config_reader import Configuration

config =Configuration().get()

SamplingFunction = Callable[[int], float | ndarray]
TargetFunction = Callable[[pd.DataFrame], pd.Series]
LinkFunction = Callable[[pd.Series], pd.Series]


@dataclass
class DistributionInterface:
    class_name: str
    required_params: list = field(default_factory=list)

    def params_valid(self, params_dict: dict) -> bool:
        if not all(params_dict.keys().__contains__(rp) for rp in self.required_params):
            raise ValueError(
                f"Required params: {self.required_params}. Provided params: {params_dict}."
            )


@dataclass
class AbstractDistribution(ABC):
    """
    A short-cut to numpy's random sampling functionalities to generate features / covariates
    """

    distribution_params: dict = field(default_factory=dict)
    interface: DistributionInterface = field(init=False, repr=False)
    seed: int = field(init=False)
    rnd_generator: Generator = field(init=False, repr=False)
    sampling_function: SamplingFunction = field(init=False, repr=False)

    def __post_init__(self):
        self.interface = self._set_interface_()

        if not self.distribution_params == {}:
            self._validate_params(self.interface, self.distribution_params)
            self._apply_interface(self.interface.required_params, self.distribution_params)
            self.seed = self._set_seed()
            self.rnd_generator = self._set_rnd_generator(self.seed)
            self.sampling_function = self._set_sampling_function()

    def _set_seed(self) -> int:
        return 1

    def _set_rnd_generator(self, seed: int) -> Generator:
        generator = Generator(PCG64(seed=seed))
        return generator

    def _apply_interface(
        self, interface_required_params: list, distribution_params: dict
    ):
        for rp in interface_required_params:
            self.__setattr__(rp, distribution_params[rp])

    def _validate_params(self, interface: DistributionInterface, params: dict) -> bool:
        return interface.params_valid(params_dict=params)

    @abstractmethod
    def _set_interface_(self) -> DistributionInterface:
        """Returns DistributionInterface that defines the required paramaters

        Returns:
            DistributionInterface: _description_
        """
        ...

    @abstractmethod
    def _set_sampling_function(
        self, params: dict, rnd_generator: Generator
    ) -> SamplingFunction:
        """_summary_

        Args:
            params (dict): _description_
            rnd_generator (Generator): _description_

        Returns:
            SamplingFunction: _description_
        """
        ...

    def sample_n(self, n: int) -> list[float]:
        return self.sampling_function(size=n)


@dataclass
class BetaDist(AbstractDistribution):
    def _set_interface_(self) -> DistributionInterface:
        return DistributionInterface("Beta", required_params=["a", "b"])

    def _set_sampling_function(self) -> SamplingFunction:
        return partial(self.rnd_generator.beta, a=self.a, b=self.b)


@dataclass
class NormalDist(AbstractDistribution):
    def _set_interface_(self) -> DistributionInterface:
        return DistributionInterface("Normal", required_params=["loc", "scale"])

    def _set_sampling_function(self) -> SamplingFunction:
        return partial(self.rnd_generator.normal, loc=self.loc, scale=self.scale)


@dataclass
class UniformDist(AbstractDistribution):
    def _set_interface_(self) -> DistributionInterface:
        return DistributionInterface("Uniform", required_params=["low", "high"])

    def _set_sampling_function(self) -> SamplingFunction:
        return partial(self.rnd_generator.uniform, low=self.low, high=self.high)


@dataclass
class DiscreteDist(AbstractDistribution):
    def _set_interface_(self) -> DistributionInterface:
        return DistributionInterface(
            "Discrete", required_params=["values", "probabilities"]
        )

    def _set_sampling_function(self) -> SamplingFunction:
        return partial(
            self.rnd_generator.choice, a=self.values, p=self.probabilities, replace=True
        )


@dataclass
class DistributionFactory(AbstractFactory):
    dist_type: Literal['Beta', 'Uniform', 'Discrete', 'Normal']
    distribution_map = {'Beta': BetaDist,
                               'Uniform': UniformDist,
                               'Discrete': DiscreteDist,
                               'Normal': NormalDist}
    distribution_class: Type[AbstractDistribution ] = field(init=False)

    
    def __post_init__(self):
        self.distribution_class = self.distribution_map[self.dist_type]

    def produce_one(self, distribution_params: tuple) -> AbstractDistribution:
        param_dict = {}
        required = self.distribution_class().interface.required_params
        for i, key in enumerate(required):
            i
            key
            param_dict[key] = distribution_params[i]
        
        return self.distribution_class(param_dict)
