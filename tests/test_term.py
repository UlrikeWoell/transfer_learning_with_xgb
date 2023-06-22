from src.data_generation.distribution import DiscreteDist
from src.data_generation.term import DistributionTerm, Generator, TermFactory


def test_term():
    term = DistributionTerm(
        coefficient=0, variable_name="X0", variable_distribution=None
    )
    assert isinstance(term, DistributionTerm)


def test_factory_generate_names():
    fact = TermFactory(variable_name_prefix="x")
    assert isinstance(fact.names, Generator)
    assert next(fact.names) == "X1"
    assert next(fact.names) == "X2"


def test_term_factory_produce_one():
    dis = DiscreteDist(distribution_params={"values": [1], "probabilities": None})
    fact = TermFactory(variable_name_prefix="x")
    term = fact.produce_one((dis, dis))
    assert isinstance(term, DistributionTerm)
    assert term.variable_name == "X1"
    assert term.coefficient == 1
    assert isinstance(term.variable_distribution, DiscreteDist)


def test_term_factory_produce_n():
    dis = DiscreteDist(distribution_params={"values": [1], "probabilities": None})
    fact = TermFactory(variable_name_prefix="x")
    terms = fact.produce_n((dis, dis), 2)
    assert all([isinstance(term, DistributionTerm) for term in terms])
    assert len(terms) == 2


def test_term_factory_produce_from_list():
    dis = DiscreteDist(distribution_params={"values": [1], "probabilities": None})
    fact = TermFactory(variable_name_prefix="x")
    input_list = [(dis, dis), (dis, dis)]
    terms = fact.produce_from_list(input_list)
    assert all([isinstance(term, DistributionTerm) for term in terms])
    assert len(terms) == 2
