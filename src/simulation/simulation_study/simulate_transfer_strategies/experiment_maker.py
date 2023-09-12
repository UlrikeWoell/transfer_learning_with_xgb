from dataclasses import asdict, dataclass
import datetime
import json
from pathlib import Path
import random

import numpy as np
from typing import Any, Dict

import pandas as pd

from src.data_generating_process.domain import (
    Domain,
    DomainGenerator,
    DomainParameters,
    DomainSampler,
    SamplingParameters,
)

class NumpyEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, np.ndarray):
            return o.tolist()
        return json.JSONEncoder.default(self, o)


DATA_DIR = "data"

@dataclass
class DataOriginLog:
    domain: Dict[str, Any]
    sample: Dict[str, Any]


@dataclass
class DataSet:
    """
    A DataSet has a dataframe with features and target variable,
    a log that
    """

    data: pd.DataFrame
    datalog: DataOriginLog
    name: str


class DataSetMaker:
    @classmethod
    def _write_csvfile(cls, data: pd.DataFrame, filepath: str, filename: str):
        """Writes a pd.DataFrame to csv

        Args:
            data (pd.DataFrame): the data to be saved in CSV
            filepath (str): file destination
            filename (str): must include ".csv"
        """
        complete_path = Path(f"{DATA_DIR}/{filepath}/{filename}")
        complete_path.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(complete_path, index=False)

    @classmethod
    def _make_data_origin_log(
        cls, domain: Domain, sample: SamplingParameters
    ) -> DataOriginLog:
        """
        Records the origin of dataset.

        Domain describes the distribution
        Sample describes the sampling procedure (seed and size)

        Args:
            domain (Domain): the Domain that created the data
            sample (Sample): the Sample that created the data

        Returns:
            DataOriginLog: origin of the dataset
        """
        domain_dict = asdict(domain)
        sample_dict = asdict(sample)
        return DataOriginLog(domain=domain_dict, sample=sample_dict)

    @classmethod
    def _write_data_origin_log_to_json(
        cls, data_origin_log: DataOriginLog, filepath: str, filename: str
    ):
        """Converts data origin log to dict and writes it to json file

        Args:
            data_origin_log (DataOriginLog): _description_
            filepath (str): destination of the json file
            filename (str): name of the json file, must include ".json"
        """
        complete_path = Path(f"{DATA_DIR}/{filepath}/{filename}")
        complete_path.parent.mkdir(parents=True, exist_ok=True)
        origin_log_dict = asdict(data_origin_log)
        with open(complete_path, "w") as json_file:
            json.dump(origin_log_dict, json_file, indent=4, cls=NumpyEncoder)

    @classmethod
    def make_dataset(
        cls, domain: Domain, sample: SamplingParameters, name: str
    ) -> DataSet:
        """Generates one dataset consisting of the data, the data_origin_log and a name

        Args:
            domain (Domain): used for data generation, will be logged in origin_log
            sample (Sample): used for data generation, will be logged in origin_log
            name (str): name of the dataset

        Returns:
            DataSet: _description_
        """
        data = DomainSampler.generate_data(domain=domain, sample=sample)
        datalog = cls._make_data_origin_log(domain=domain, sample=sample)
        return DataSet(data=data, datalog=datalog, name=name)

    @classmethod
    def materialize_dataset(cls, dataset: DataSet, filepath: str):
        """The data is saved in a csv file. The data_origin_log is saved in a json.

        Args:
            dataset (DataSet): dataset with data, origin_log and name
            filepath (str): where both files will be saved
        """
        cls._write_csvfile(
            data=dataset.data, filepath=filepath, filename=f"{dataset.name}.csv"
        )
        cls._write_data_origin_log_to_json(
            data_origin_log=dataset.datalog,
            filepath=filepath,
            filename=f"{dataset.name}_log.json",
        )


class TransferExperimentMaker:
    """
    Creates 4 files for a transfer learning experiment and saves them on the disk
        src_train, src_test, tgt_train, tgt_test

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
                self.src_fixed_parameters, domain_seed
            ),
            "tgt": DomainGenerator().get_domain(
                self.tgt_fixed_parameters, domain_seed
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
                # check if generated data contains samples of both classes
                if (0 in dataset.data["y"].values) and (1 in dataset.data["y"].values):
                    datasets.append(dataset)

        # if any file did not contain samples of both classes,
        # skip this dataset and don't save it
        if len(datasets) == 4:
            for data in datasets:
                DataSetMaker.materialize_dataset(dataset=data, filepath=filepath)
        else:
            print("invalid dataset")


