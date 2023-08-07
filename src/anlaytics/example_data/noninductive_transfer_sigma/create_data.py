import datetime
import random
import time

from src.data_generating_process.data_set_maker import DataSetMaker
from src.data_generating_process.domain_parameters import (
    RndCorrellationMatrix,
    CensoredVariables,
    BernoulliBias
)
from src.data_generating_process.domainV2 import (
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
    n_seeds = 2
    domain_seeds = [start_seed + i * 100 for i in range(n_seeds)]
    return domain_seeds


def get_domain_pair(pair_seed: int):
    base_coeff_cnt = 5
    p1 = DomainParameters(
        base_coeff_cnt=base_coeff_cnt,
        correllation_matrix=RndCorrellationMatrix(base_coeff_cnt, seed=pair_seed + 1),
        censored_variables=CensoredVariables([]),
        intr_coefficients_prob_of_zero=0,
        base_coefficients_prob_of_zero=0,
        bias=BernoulliBias(0)
    )

    p2 = DomainParameters(
        base_coeff_cnt=base_coeff_cnt,
        correllation_matrix=RndCorrellationMatrix(base_coeff_cnt, seed=pair_seed + 2),
        censored_variables=CensoredVariables([]),
        intr_coefficients_prob_of_zero=0,
        base_coefficients_prob_of_zero=0,
        bias=BernoulliBias(0)
    )

    d1 = DomainGenerator.get_domain(p1, seed=pair_seed)
    d2 = DomainGenerator.get_domain(p2, seed=pair_seed)
    return {"source": d1, "target": d2}


def get_sample_sizes():
    return {"source": 2000, "target": 200}


def get_sample_seeds():
    return {
        "source_train": 241262023,
        "source_test": 251120923,
        "target_train": 12192092,
        "target_test": 102437538,
    }


def get_file_path():
    return f"noninductive_transfer/sigma/{get_timestamp()}"


def create_data(pair_seed: int):
    domains = get_domain_pair(pair_seed)
    sample_sizes = get_sample_sizes()
    sample_seeds = get_sample_seeds()
    filepath = get_file_path()

    # Sampline parameters
    s_sp_train = SamplingParameters(
        sample_sizes["source"], sample_seeds["source_train"]
    )
    s_sp_test = SamplingParameters(500, sample_seeds["source_test"])
    t_sp_train = SamplingParameters(
        sample_sizes["target"], sample_seeds["target_train"]
    )
    t_sp_test = SamplingParameters(500, sample_seeds["target_test"])

    # Data sets
    s_train = DataSetMaker.make_dataset(
        domain=domains["source"], sample=s_sp_train, name="s_train"
    )
    s_test = DataSetMaker.make_dataset(
        domain=domains["source"], sample=s_sp_test, name="s_test"
    )
    t_train = DataSetMaker.make_dataset(
        domain=domains["target"], sample=t_sp_train, name="t_train"
    )
    t_test = DataSetMaker.make_dataset(
        domain=domains["target"], sample=t_sp_test, name="t_test"
    )

    # Save to file
    DataSetMaker.materialize_dataset(dataset=s_train, filepath=filepath)
    DataSetMaker.materialize_dataset(dataset=s_test, filepath=filepath)
    DataSetMaker.materialize_dataset(dataset=t_train, filepath=filepath)
    DataSetMaker.materialize_dataset(dataset=t_test, filepath=filepath)


start_seed = 30303
n_pairs = 30
pair_seeds = range(start_seed, start_seed + n_pairs, 1)

for s in pair_seeds:
    create_data(s)
