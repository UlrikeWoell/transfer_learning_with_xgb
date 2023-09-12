import math

from src.anlaytics.simulation_study.simulate_transfer_strategies.experiment_maker import (
    TransferExperimentMaker,
)
from src.data_generating_process.domain import DomainParameters
from src.data_generating_process.domain_parameters import (
    RndBernoulliBias,
    RndCensoredVariables,
    RndCoefficients,
    RndCorrelationMatrix,
    RndTransformationExponent,
)

NUMBER_OF_EXPERIMENTS = 5

SRC_TRAIN_SIZE = 2000
SRC_TEST_SIZE = 500
TGT_TRAIN_SIZE = 200
TGT_TEST_SIZE = 500

FIRST_SAMPLE_SEED = 9799
FIRST_SHARED_DOMAIN_SEED = 13888
FIRST_SEED_SRC_FIXTURE = 7477
FIRST_SEED_TGT_FIXTURE = 80303

NUMBER_OF_RND_CENSORED_FEATURES = 6
NUMBER_OF_BASE_FEATURES = 30
PROB_OF_ZERO_BASE_COEFF = 0.8
VALUE_RANGE_BASE_COEFF = 10
PROB_OF_ZERO_INTERACTION_COEFF = 0.5
VALUE_RANGE_INTERACTION_COEFF = 10
CORRELATION_VALUE_RANGE = 0.5
BIAS_VALUE_RANGE = 0.3
EXPONENT_LOWER = 0.5
EXPONENT_UPPPER = 2


class FixedParams:
    def __init__(
        self,
        start_seed: int,
        base_coeff_count: int,
        size_censored: int,
        number_of_experiments: int,
    ) -> None:
        self.base_coef_count = base_coeff_count
        self.size_censored = size_censored
        self.rnd_seeds = [
            1000 * start_seed + i + i for i in range(0, number_of_experiments, 1)
        ]

    def get_matrices_same_sign(self):
        fixed = [
            RndCorrelationMatrix(
                self.base_coef_count, seed, True, CORRELATION_VALUE_RANGE
            )
            for seed in self.rnd_seeds
        ]
        parameters = [
            DomainParameters(base_coeff_cnt=self.base_coef_count, correllation_matrix=x)
            for x in fixed
        ]
        return parameters

    def get_matrices_any_sign(self):
        fixed = [
            RndCorrelationMatrix(self.base_coef_count, seed, False, CORRELATION_VALUE_RANGE)
            for seed in self.rnd_seeds
        ]
        parameters = [
            DomainParameters(base_coeff_cnt=self.base_coef_count, correllation_matrix=x)
            for x in fixed
        ]
        return parameters

    def get_censored_variables(self):
        fixed = [
            RndCensoredVariables(i, self.size_censored, self.base_coef_count)
            for i in self.rnd_seeds
        ]
        parameters = [
            DomainParameters(base_coeff_cnt=self.base_coef_count, censored_variables=x)
            for x in fixed
        ]
        return parameters

    def get_biases(self):
        fixed = [RndBernoulliBias(i, BIAS_VALUE_RANGE) for i in self.rnd_seeds]
        parameters = [
            DomainParameters(base_coeff_cnt=self.base_coef_count, bias=x) for x in fixed
        ]
        return parameters

    def get_exponents(self):
        fixed = [
            RndTransformationExponent(
                i,
                exponent_range_lower=EXPONENT_LOWER,
                exponent_range_upper=EXPONENT_UPPPER,
            )
            for i in self.rnd_seeds
        ]
        parameters = [
            DomainParameters(
                base_coeff_cnt=self.base_coef_count, transformation_exponent=x
            )
            for x in fixed
        ]
        return parameters

    def get_coeffs_any_sign(self):
        fixed_base = [
            RndCoefficients(
                size=self.base_coef_count,
                seed=i,
                prob_of_zero=PROB_OF_ZERO_BASE_COEFF,
                same_sign=False,
                value_range=VALUE_RANGE_BASE_COEFF,
            )
            for i in self.rnd_seeds
        ]
        fixed_intr = [
            RndCoefficients(
                size=math.comb(self.base_coef_count, 2),
                seed=i + 45,
                prob_of_zero=PROB_OF_ZERO_INTERACTION_COEFF,
                same_sign=False,
                value_range=VALUE_RANGE_INTERACTION_COEFF,
            )
            for i in self.rnd_seeds
        ]

        parameters = [
            DomainParameters(
                base_coeff_cnt=self.base_coef_count,
                base_coefficients=b,
                intr_coefficients=i,
            )
            for b, i in zip(fixed_base, fixed_intr)
        ]

        return parameters

    def get_coeffs_same_sign(self):
        fixed_base = [
            RndCoefficients(
                size=self.base_coef_count,
                seed=i,
                prob_of_zero=PROB_OF_ZERO_BASE_COEFF,
                same_sign=True,
                value_range=VALUE_RANGE_BASE_COEFF,
            )
            for i in self.rnd_seeds
        ]

        fixed_intr = [
            RndCoefficients(
                size=math.comb(self.base_coef_count, 2),
                seed=i + 45,
                prob_of_zero=PROB_OF_ZERO_INTERACTION_COEFF,
                same_sign=True,
                value_range=VALUE_RANGE_INTERACTION_COEFF,
            )
            for i in self.rnd_seeds
        ]

        parameters = [
            DomainParameters(
                base_coeff_cnt=self.base_coef_count,
                base_coefficients=b,
                intr_coefficients=i,
            )
            for b, i in zip(fixed_base, fixed_intr)
        ]

        return parameters


def create_bulk_data():
    src_fixture = FixedParams(
        FIRST_SEED_SRC_FIXTURE,
        NUMBER_OF_BASE_FEATURES,
        NUMBER_OF_RND_CENSORED_FEATURES,
        NUMBER_OF_EXPERIMENTS,
    )
    tgt_fixture = FixedParams(
        FIRST_SEED_TGT_FIXTURE,
        NUMBER_OF_BASE_FEATURES,
        NUMBER_OF_RND_CENSORED_FEATURES,
        NUMBER_OF_EXPERIMENTS,
    )

    # Prepare the scenarios.
    # For each scenario, get pairs of different parameters.
    parameters = {
        "matrix_same_sign": list(
            zip(
                src_fixture.get_matrices_same_sign(),
                tgt_fixture.get_matrices_same_sign(),
            )
        ),
        "matrix_any_sign": list(
            zip(
                src_fixture.get_matrices_any_sign(), tgt_fixture.get_matrices_any_sign()
            )
        ),
        "bias": list(zip(src_fixture.get_biases(), tgt_fixture.get_biases())),
        "censor": list(
            zip(
                src_fixture.get_censored_variables(),
                tgt_fixture.get_censored_variables(),
            )
        ),
        "coeffs_same_sign": list(
            zip(src_fixture.get_coeffs_same_sign(), tgt_fixture.get_coeffs_same_sign())
        ),
        "coeffs_any_sign": list(
            zip(src_fixture.get_coeffs_any_sign(), tgt_fixture.get_coeffs_any_sign())
        ),
        "exponents": list(
            zip(src_fixture.get_exponents(), tgt_fixture.get_exponents())
        ),
    }

    # For all parameter pairs in all scenarios, create data
    # from a domain that is identical except for the one parameter pair
    for i, (scenario, param_pairs) in enumerate(parameters.items()):
        for j, pair in enumerate(param_pairs):
            print(f"{i}.{scenario}: {j}/{len(param_pairs)}")
            tdgp = TransferExperimentMaker(
                src_train_size=SRC_TRAIN_SIZE,
                src_test_size=SRC_TEST_SIZE,
                tgt_train_size=TGT_TRAIN_SIZE,
                tgt_test_size=TGT_TEST_SIZE,
                sample_seed=FIRST_SAMPLE_SEED * (i + 1) * (j + 1),
                src_fixed_parameters=pair[1],
                tgt_fixed_parameters=pair[0],
                shared_domain_seed=FIRST_SHARED_DOMAIN_SEED * (i + 1) * (j + 1),
                save_files_at=f"simulation/{scenario}",
            )
            tdgp.create_data()

    print("Done")


create_bulk_data()
