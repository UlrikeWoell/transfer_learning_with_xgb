import datetime
import random
import time

from src.anlaytics.simulation_study.simulate_transfer_strategies.experiment_maker import DataSetMaker
from src.data_generating_process.domain_parameters import RndCensoredVariables
from src.data_generating_process.domain import (
    Domain,
    DomainGenerator,
    DomainParameters,
    SamplingParameters,
)


class SimpleDGP:
    """
    Use Simple DGP to create datasets from domains with common parameter.


    Example usage:
    ===============

    params = DomainParameters(5)

    dgp = SimpleDGP(
        n_sample_seeds=1,
        n_domain_seeds=3,
        constant_domain_parameters=params,
        save_files_at="data/example",
        sample_size=300,
    )

    dgp.create_data()

    """

    def __init__(
        self,
        n_sample_seeds: int,
        n_domain_seeds: int,
        constant_domain_parameters: DomainParameters,
        save_files_at: str,
        sample_size: int,
    ) -> None:
        self.n_domain_seeds = n_domain_seeds
        self.constant_domain_parameters = constant_domain_parameters
        self.n_sample_seeds = n_sample_seeds
        self.sample_size = sample_size
        self.save_files_at = save_files_at

    def get_unique_name(self):
        """Utility method to make a name for files or folders
        from the current timestamp and a random number"""
        return "_".join(
            [
                datetime.datetime.now().strftime("%m%d%H%M%S%s"),
                str(random.randint(100, 999)),
            ]
        )

    def _get_domain_seeds(self):
        start_seed = 12061984
        domain_seeds = [start_seed + i * 100 for i in range(self.n_domain_seeds)]
        return domain_seeds

    def _get_domains(self) -> list[Domain]:
        domain_seeds = self._get_domain_seeds()
        return [
            DomainGenerator().get_domain(params=self.constant_domain_parameters, seed=s)
            for s in domain_seeds
        ]

    def _get_sample_seeds(self):
        start_seed = 5688745764
        sample_seeds = [start_seed + i * 100 for i in range(self.n_sample_seeds)]
        return sample_seeds

    def _get_file_path(self):
        return f"{self.save_files_at}"

    def create_data(self):
        """Writes the desired csv files to the given folder"""
        domains = self._get_domains()
        sample_size = self.sample_size
        sample_seeds = self._get_sample_seeds()

        for d in domains:
            for ss in sample_seeds:
                filepath = self._get_file_path()
                sp = SamplingParameters(sample_size, ss)
                train = DataSetMaker.make_dataset(
                    domain=d, sample=sp, name=self.get_unique_name()
                )
                DataSetMaker.materialize_dataset(dataset=train, filepath=filepath)


