from pandas import DataFrame, Series

from src.data_generation.data_generator import (
    DataGenerationStrategy,
    DataGenerator,
    additive_target_fct,
    logit_random_link_fct,
)
from src.data_generation.distribution import DiscreteDist
from src.data_generation.term import Term, TermFactory


def test_additive_target_fct():
    assert True


def test_strategy():
    dist = DiscreteDist(distribution_params={"values": [1], "probabilities": None})

    terms = TermFactory("x").produce_n((dist, dist), 2)

    strat = DataGenerationStrategy(
        terms=terms, target_fct=additive_target_fct, link_fct=logit_random_link_fct
    )
    assert isinstance(strat, DataGenerationStrategy)


def test_generator_generate_data():
    dist = DiscreteDist(distribution_params={"values": [1], "probabilities": None})
    terms = TermFactory("x").produce_n((dist, dist), 2)
    strat = DataGenerationStrategy(
        terms=terms, target_fct=additive_target_fct, link_fct=logit_random_link_fct
    )

    dgp = DataGenerator()
    features, target, transformed_target = dgp.generate_data(strat, 3)
    assert isinstance(features, DataFrame)
    assert isinstance(target, Series)
    assert isinstance(transformed_target, Series)