import itertools as it
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import partial
from typing import Callable, List, Literal

@dataclass
class AbstractFactory(ABC):
    @abstractmethod
    def produce_one(self, input:object):
        ...

    def produce_from_list(self, input_list: list):
        return[self.produce_one(line) for line in input_list]

    def produce_n(self, input: object, n: int):
        input_list = [input for i in range(n)]
        return self.produce_from_list(input_list)