import datetime
import random
import time

from src.anlaytics.study.simulate_transfer_strategies.experiment_maker import DataSetMaker
from src.data_generating_process.domain_parameters import RndCensoredVariables
from src.data_generating_process.domain import (
    DomainGenerator,
    DomainParameters,
    SamplingParameters,
)


def get_timestamp():
    """Utility method to get current timestamp for filenames"""
    return "_".join(
        [
            datetime.datetime.now().strftime("%m%d%H%M%S%s"),
            str(random.randint(100, 999)),
        ]
    )


def get_domain_seeds():
    start_seed = 12061984
    n_seeds = 20
    domain_seeds = [start_seed + i * 100 for i in range(n_seeds)]
    return domain_seeds


def get_domains():
    domain_seeds = get_domain_seeds()
    base_coeff_cnt = 30

    domains = [
        DomainGenerator().get_domain(
            params=DomainParameters(
                base_coeff_cnt=base_coeff_cnt,
                base_coefficients_prob_of_zero=0.8,
                intr_coefficients_prob_of_zero=0.5,
                censored_variables=RndCensoredVariables(
                    censored_size=round(base_coeff_cnt * 0.2),
                    total_size=base_coeff_cnt,
                    seed=s + 1,
                ),
            ),
            seed=s,
        )
        for s in domain_seeds
    ]
    return domains


def get_sample_sizes():
    return [i for i in range(100, 3000, 100)]


def get_sample_seeds():
    return [12121964, 28051966]


def get_file_path():
    return f"validate_dgp/{get_timestamp()}"


def create_data():
    domains = get_domains()
    sample_sizes = get_sample_sizes()
    train_sample_seeds = get_sample_seeds()[0]
    test_sample_seed = get_sample_seeds()[1]

    for d in domains:
        for n in sample_sizes:
            filepath = get_file_path()
            sp_train = SamplingParameters(n, train_sample_seeds)
            sp_test = SamplingParameters(500, test_sample_seed)
            train = DataSetMaker.make_dataset(domain=d, sample=sp_train, name="train")
            test = DataSetMaker.make_dataset(domain=d, sample=sp_test, name="test")
            DataSetMaker.materialize_dataset(dataset=train, filepath=filepath)
            DataSetMaker.materialize_dataset(dataset=test, filepath=filepath)
            print(filepath, n)


create_data()
