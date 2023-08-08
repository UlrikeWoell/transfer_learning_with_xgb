import datetime
import random
import time

from src.data_generating_process.data_set_maker import DataSetMaker
from src.data_generating_process.domain_parameters import RndCensoredVariables
from src.data_generating_process.domainV2 import (
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


class TransferDGP:
    """
    Creates a set of 4 files: src_train, src_test, tgt_train, tgt_test
    Example usage:
    ==============

    for i in [3,4,5]:
        params = DomainParameters(3)
        tdgp = TransferDGP(
            tgt_train_size=20,
            tgt_test_size=32,
            src_test_size=332,
            src_train_size=32,
            domain_seed=123,
            sample_seed=234,
            constant_domain_parameters=params,
            save_files_at="data/example/transfer",
        )
        tdgp.create_data()
    """

    def __init__(
        self,
        tgt_train_size: int,
        tgt_test_size: int,
        src_train_size: int,
        src_test_size: int,
        sample_seed: int,
        constant_domain_parameters: DomainParameters,
        domain_seed: int,
        save_files_at: str,
    ) -> None:
        self.tgt_train_size = tgt_train_size
        self.tgt_test_size = tgt_test_size
        self.src_train_size = src_train_size
        self.src_test_size = src_test_size
        self.sample_seed = sample_seed
        self.constant_domain_parameters = constant_domain_parameters
        self.domain_seed = domain_seed
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
        return {"src": self.domain_seed, "tgt": self.domain_seed + 1}

    def _get_domains(self) -> dict[str, Domain]:
        domain_seeds = self._get_domain_seeds()
        return {
            "src": DomainGenerator().get_domain(
                self.constant_domain_parameters, domain_seeds["src"]
            ),
            "tgt": DomainGenerator().get_domain(
                self.constant_domain_parameters, domain_seeds["tgt"]
            ),
        }

    def _get_sample_seeds(self) -> dict[str, int]:
        return {
            "src_train": self.sample_seed + 1,
            "src_test": self.sample_seed + 2,
            "tgt_train": self.sample_seed + 3,
            "tgt_test": self.sample_seed + 4,
        }

    def _get_sample_sizes(self):
        return {
            "src_train": self.src_train_size,
            "src_test": self.src_test_size,
            "tgt_train": self.tgt_train_size,
            "tgt_test": self.tgt_test_size,
        }

    def _get_file_path(self):
        return f"{self.save_files_at}/{self.get_unique_name()}"

    def create_data(self):
        """Writes the desired csv files to the given folder"""
        domains = self._get_domains()
        sample_sizes = self._get_sample_sizes()
        sample_seeds = self._get_sample_seeds()

        filepath = self._get_file_path()
        for x in ["src", "tgt"]:
            for y in ["train", "test"]:
                sp = SamplingParameters(
                    sample_sizes[f"{x}_train"], sample_seeds[f"{x}_{y}"]
                )
                data = DataSetMaker.make_dataset(
                    domain=domains[x], sample=sp, name=f"{x}_{y}"
                )

                DataSetMaker.materialize_dataset(dataset=data, filepath=filepath)
