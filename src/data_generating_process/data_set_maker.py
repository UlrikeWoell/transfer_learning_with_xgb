import datetime
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.data_generating_process.domainV2 import (
    Domain,
    DomainGenerator,
    DomainParameters,
    DomainSampler,
    SamplingParameters,
)

DATA_DIR = "data"


class NumpyEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, np.ndarray):
            return o.tolist()
        return json.JSONEncoder.default(self, o)


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

