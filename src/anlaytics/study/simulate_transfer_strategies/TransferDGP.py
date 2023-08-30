import datetime
import random

from src.data_generating_process.data_set_maker import DataSetMaker
from src.data_generating_process.domainV2 import (
    Domain,
    DomainGenerator,
    DomainParameters,
    SamplingParameters,
)


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
        src_fixed_parameters: DomainParameters,
        tgt_fixed_parameters: DomainParameters,
        shared_domain_seed: int,
        save_files_at: str,
    ) -> None:
        self.tgt_train_size = tgt_train_size
        self.tgt_test_size = tgt_test_size
        self.src_train_size = src_train_size
        self.src_test_size = src_test_size
        self.sample_seed = sample_seed
        self.src_fixed_parameters = src_fixed_parameters
        self.tgt_fixed_parameters = tgt_fixed_parameters
        self.shared_domain_seed = shared_domain_seed
        self.save_files_at = save_files_at

    def get_unique_name(self):
        """Utility method to make a name for files or folders
        from the current timestamp and a random number"""

        return "_".join(
            [
                str(random.randint(100, 999)),
                datetime.datetime.now().strftime("%m%d%H%M%S%s"),
            ]
        )

    def _get_domain_seeds(self):
        return self.shared_domain_seed

    def _get_domains(self) -> dict[str, Domain]:
        domain_seed = self._get_domain_seeds()
        return {
            "src": DomainGenerator().get_domain(
                self.src_fixed_parameters, domain_seed, matrix_same_sign=True
            ),
            "tgt": DomainGenerator().get_domain(
                self.tgt_fixed_parameters, domain_seed, matrix_same_sign=True
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
        datasets = []
        for x in ["src", "tgt"]:
            for y in ["train", "test"]:
                sp = SamplingParameters(
                    sample_sizes[f"{x}_{y}"], sample_seeds[f"{x}_{y}"]
                )
                dataset = DataSetMaker.make_dataset(
                    domain=domains[x], sample=sp, name=f"{x}_{y}"
                )
                if (0 in dataset.data["y"].values) and (1 in dataset.data["y"].values):
                    datasets.append(dataset)

        if len(datasets) == 4:
            for data in datasets:
                DataSetMaker.materialize_dataset(dataset=data, filepath=filepath)
        else:
            print("invalid dataset")
