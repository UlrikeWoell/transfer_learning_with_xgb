import math
import os
from pprint import pprint

from src.anlaytics.study.simulate_transfer_strategies.TransferDGP import TransferDGP
from src.data_generating_process.domain_parameters import (
    RndBernoulliBias,
    RndCensoredVariables,
    RndCoefficients,
    RndCorrellationMatrix,
    RndTransformationExponent,
)
from src.data_generating_process.domainV2 import DomainParameters

NUMBER_OF_EXPERIMENTS = 100


class FixedParams:
    def __init__(self, start_seed) -> None:
        self.base_coef_count = 30
        self.size_censored = 6
        self.rnd_seeds = [1000 * start_seed + i+i for i in range(50,100,1)]

    def get_matrices_same_sign(self):
        fixed = [
            RndCorrellationMatrix(self.base_coef_count, seed, True)
            for seed in self.rnd_seeds
        ]
        parameters = [
            DomainParameters(base_coeff_cnt=self.base_coef_count, correllation_matrix=x)
            for x in fixed
        ]
        return parameters

    def get_matrices_any_sign(self):
        fixed = [
            RndCorrellationMatrix(self.base_coef_count, seed, False)
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
        fixed = [RndBernoulliBias(i) for i in self.rnd_seeds]
        parameters = [
            DomainParameters(base_coeff_cnt=self.base_coef_count, bias=x) for x in fixed
        ]
        return parameters

    def get_exponents(self):
        fixed = [RndTransformationExponent(i) for i in self.rnd_seeds]
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
                prob_of_zero=0.8,
                same_sign=False,
                value_range=10,
            )
            for i in self.rnd_seeds
        ]

        fixed_intr = [
            RndCoefficients(
                size=math.comb(self.base_coef_count, 2),
                seed=i + 45,
                prob_of_zero=0.5,
                same_sign=False,
                value_range=10,
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
                prob_of_zero=0.8,
                same_sign=True,
                value_range=10,
            )
            for i in self.rnd_seeds
        ]

        fixed_intr = [
            RndCoefficients(
                size=math.comb(self.base_coef_count, 2),
                seed=i + 45,
                prob_of_zero=0.5,
                same_sign=True,
                value_range=10,
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
# Create the data files
    src_fixture = FixedParams(7477)
    tgt_fixture = FixedParams(80303)

    parameters = {
        "matrix_same_sign": list(
            zip(src_fixture.get_matrices_same_sign(), tgt_fixture.get_matrices_same_sign())
        ),
        "matrix_any_sign": list(
            zip(src_fixture.get_matrices_any_sign(), tgt_fixture.get_matrices_any_sign())
        ),
        "bias": list(zip(src_fixture.get_biases(), tgt_fixture.get_biases())),
        "censor": list(
            zip(src_fixture.get_censored_variables(), tgt_fixture.get_censored_variables())
        ),
        "coeffs_same_sign": list(
            zip(src_fixture.get_coeffs_same_sign(), tgt_fixture.get_coeffs_same_sign())
        ),
        "coeffs_any_sign": list(
            zip(src_fixture.get_coeffs_any_sign(), tgt_fixture.get_coeffs_any_sign())
        ),
        "exponents": list(zip(src_fixture.get_exponents(), tgt_fixture.get_exponents())),
    }

    for i, (scenario, param_pairs) in enumerate(parameters.items()):
        for j, pair in enumerate(param_pairs):
            print(f'{i}.{scenario}: {j}/{len(param_pairs)}')
            tdgp = TransferDGP(
                src_train_size=2000,
                src_test_size=500,
                tgt_train_size=200,
                tgt_test_size=500,
                sample_seed=9799 * (i + 1) * (j + 1),
                src_fixed_parameters=pair[1],
                tgt_fixed_parameters=pair[0],
                shared_domain_seed=13888 * (i + 1) * (j + 1),
                save_files_at=f"simulation/{scenario}",
            )
            tdgp.create_data()

    print("Done")




create_bulk_data()