from dataclasses import asdict, dataclass, field
from time import time

from src.domains.domains import Domain, DomainChanger
from src.transfer_strategies.transfer_strategies import (
    CONTINUE,
    REVISE,
    SRCONLY,
    TGTONLY,
    WEIGHTED,
    AbstractTransferStrategy,
    SamplingTask,
    SimulationSetUp,
)
from src.util.config_reader import Configuration


@dataclass
class Run:
    strategy: AbstractTransferStrategy
    domain_seed: int
    n_features: int
    src_sampling_task: SamplingTask
    tgt_sampling_task: SamplingTask
    tgt_modification_type: str
    tgt_modification_factor: float


def make_runs():
    c = Configuration().get()
    src_dom_seeds = c["SOURCE_DOMAIN_CONFIG"]["src_domain_seeds"]
    src_dom_n_feats = c["SOURCE_DOMAIN_CONFIG"]["src_domain_n_features"]

    covariance_factors = c["DOMAIN_CHANGE_CONFIG"]["covariance_factors"]
    noise_variance_factors = c["DOMAIN_CHANGE_CONFIG"]["noise_variance_factors"]
    bernoulli_bias_factors = c["DOMAIN_CHANGE_CONFIG"]["bernoulli_bias_factors"]
    coefficient_factors = c["DOMAIN_CHANGE_CONFIG"]["coefficient_factors"]
    censoring_probabilities = c["DOMAIN_CHANGE_CONFIG"]["censoring_probabilities"]

    src_train_sample_seeds = c["SAMPLING_SEEDS"]["src_train_sample_seeds"]
    src_test_sample_seeds = c["SAMPLING_SEEDS"]["src_test_sample_seeds"]
    tgt_train_sample_seeds = c["SAMPLING_SEEDS"]["tgt_train_sample_seeds"]
    tgt_test_sample_seeds = c["SAMPLING_SEEDS"]["src_train_sample_seeds"]

    src_train_sample_n = c["SAMPLING_SIZES"]["src_train_sample_n"]
    src_test_sample_n = c["SAMPLING_SIZES"]["src_test_sample_n"]
    tgt_train_sample_n = c["SAMPLING_SIZES"]["tgt_train_sample_n"]
    tgt_test_sample_n = c["SAMPLING_SIZES"]["tgt_test_sample_n"]

    strategies = [TGTONLY, SRCONLY, WEIGHTED, REVISE, CONTINUE]

    