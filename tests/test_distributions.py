import pytest

from src.data_generation.distribution import (
    AbstractDistribution,
    BetaDist,
    DiscreteDist,
    DistributionInterface,
    Generator,
    NormalDist,
    SamplingFunction,
    UniformDist,
    partial,
    DistributionFactory
)


@pytest.fixture
def distribution():
    return 1


def test_interface():
    di = DistributionInterface("A Name", required_params=["a", "b", "c"])
    # does not raise error
    di.params_valid({"a": 1, "b": 2, "c": None})

    # does raise error
    with pytest.raises(ValueError):
        di.params_valid({"a": 1})


def test_distribution_apply_interface():
    class AbsDis(AbstractDistribution):
        def _set_interface_(self) -> DistributionInterface:
            return DistributionInterface("A Name", required_params=["low", "high"])

        def _set_sampling_function(
            self
        ) -> SamplingFunction:
            return partial(self.rnd_generator.uniform, low=self.low, high=self.high)

    absdis = AbsDis(distribution_params={"low": 1, "high": 2})
    assert isinstance(absdis, AbsDis)
    assert absdis.low == 1
    assert absdis.high == 2



distributions = [BetaDist, UniformDist, NormalDist, DiscreteDist]
params = [
    {"a": 1, "b": 2},
    {"low": 1, "high": 2},
    {"loc": 1, "scale": 2},
    {"values": [1, 3, 5], "probabilities": None},
]


@pytest.mark.parametrize("D, params", zip(distributions, params))
def test_concrete_distributions(D: callable, params: dict):
    D(params)
    assert True

wrong_params = [
    {"xa": 1, "b": 2},
    {"xlow": 1, "high": 2},
    {"xloc": 1, "scale": 2},
    {"xvalues": [1, 3, 5], "probabilities": None},
]
@pytest.mark.parametrize("D, params", zip(distributions, wrong_params))
def test_validate_concrete_distributions(D: callable, params: dict):
    with pytest.raises(ValueError):
        D(params)


def test_distribution_factory_produce_one():
    fac = DistributionFactory('Beta')
    dist =  fac.produce_one((1,2))
    assert isinstance(dist, BetaDist)
    assert len(dist.sample_n(2)) == 2


def test_distribution_factory_produce_n():
    fac = DistributionFactory('Beta')
    dists =  fac.produce_n((1,2), 2)
    assert len(dists) == 2
    assert all(isinstance(d,BetaDist) for d in dists)


def test_distribution_factory_produce_from_list():
    fac = DistributionFactory('Beta')
    input_list = [(1,2), (3,4)]
    dists =  fac.produce_from_list(input_list)
    assert len(dists) == 2
    assert all(isinstance(d,BetaDist) for d in dists)
