import math
import os

from src.anlaytics.illustration_of_datasets.SimpleDGP import SimpleDGP
from src.anlaytics.illustration_of_datasets.viz_example_data import (
    PairPlotMaker,
    PlotCombiner,
)
#from src.anlaytics.study.simulation_data.TransferDGP import TransferDGP
from src.data_generating_process.domain_parameters import (
    BernoulliBias,
    RndCensoredVariables,
    RndCoefficients,
    RndCorrelationMatrix,
    TransformationExponent,
)
from src.data_generating_process.domain import DomainParameters


class Illustrations:
    def __init__(self) -> None:
        self.base_coef_count = 5
        self.rnd_seeds = [1, 2, 3]
        self.biases = [-0.2, 0, 0.2]
        self.exponents = [0.5, 1, 1.5]
        self.domain_seed = 4598548

    def get_matrices(self):
        fixed = [
            RndCorrelationMatrix(self.base_coef_count, seed,same_sign=False,value_range=0.7) for seed in self.rnd_seeds
        ]
        parameters = [
            DomainParameters(base_coeff_cnt=self.base_coef_count, correllation_matrix=x)
            for x in fixed
        ]
        return parameters

    def get_censored_variables(self):
        fixed = [
            RndCensoredVariables(i, 2, self.base_coef_count) for i in self.rnd_seeds
        ]
        parameters = [
            DomainParameters(base_coeff_cnt=self.base_coef_count, censored_variables=x)
            for x in fixed
        ]
        return parameters

    def get_biases(self):
        fixed = [BernoulliBias(i) for i in self.biases]
        parameters = [
            DomainParameters(base_coeff_cnt=self.base_coef_count, bias=x) for x in fixed
        ]
        return parameters

    def get_exponents(self):
        fixed = [TransformationExponent(i) for i in self.exponents]
        parameters = [
            DomainParameters(
                base_coeff_cnt=self.base_coef_count, transformation_exponent=x
            )
            for x in fixed
        ]
        return parameters

    def get_coeffs(self):
        fixed_base = [
            RndCoefficients(self.base_coef_count, i, prob_of_zero=0.5, same_sign=False)
            for i in self.rnd_seeds
        ]

        fixed_intr = [
            RndCoefficients(
                math.comb(self.base_coef_count, 2), i + 45, prob_of_zero=0.5, same_sign=False
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


# Create the data files
illu = Illustrations()

parameters = {
    "matrix": illu.get_matrices(),
    "bias": illu.get_biases(),
    "censor": illu.get_censored_variables(),
    "coeffs": illu.get_coeffs(),
    "exponents": illu.get_exponents(),
}


for k, v in parameters.items():
    for params in v:
        sdgp = SimpleDGP(
            n_sample_seeds=1,
            n_domain_seeds=1,
            constant_domain_parameters=params,
            save_files_at=f"illustrations/{k}",
            sample_size=300,
        )
        sdgp.create_data()


# get the csv and json files
directory = "data/illustrations"
scenarios = sorted(os.listdir(directory))

file_pairs = {}
for s in scenarios:
    files = sorted(os.listdir(f"{directory}/{s}"))

    csv = [f for f in files if f.endswith("csv")]
    log = [f for f in files if f.endswith("json")]
    names = [f.removesuffix(".csv") for f in files if f.endswith("csv")]
    csv_log = list(zip(names, csv, log))
    file_pairs[s] = csv_log


# make the pair plot
for s, pairs in file_pairs.items():
    print(f"Plotting {s}")
    single_plots = []
    for pair in pairs:
        plotpath = f"images/illustrations/{s}/{pair[0]}.png"
        single_plots.append(plotpath)
        pm = PairPlotMaker(
            root_dir=f"data/illustrations/{s}",
            data_csv_name=pair[1],
            log_name=pair[2],
            save_path=plotpath,
            figsize=(10, 10),
        )
        pm.make_pairplot(scenario=s)

    pc = PlotCombiner()
    pc.paste_images_horizontally(
        single_plots, save_at=f"images/illustrations/{s}/3pairplots.png"
    )

print("Done")
