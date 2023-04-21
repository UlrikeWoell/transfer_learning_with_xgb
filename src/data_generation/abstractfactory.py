"""AbstractFactory to produce 1, n of the same, or n different objects"""
from abc import ABC, abstractmethod
from src.util.logger import log_debug


class AbstractFactory(ABC):
    """Produces things
    """

    @abstractmethod
    def produce_one(self, data:object):
        """
        Produces exactly one thing
        """

    def produce_from_list(self, data_list: list):
        """
        Produces one thing for each entry in input list.
        """
        log_debug(f'produce_from_list: {data_list}')
        return[self.produce_one(line) for line in data_list]

    def produce_n(self, data: object, number_of_things: int):
        """
        Produce the same thing n times
        """
        log_debug(f'produce_n with n={number_of_things}: {data}')
        return [self.produce_one(data) for i in range(number_of_things)]