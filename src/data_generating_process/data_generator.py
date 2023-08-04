import datetime
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.data_generating_process.domainV2 import Domain, Sample, SampleGenerator

DATA_FOLDER = "data"


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


@dataclass
class DataSet:
    data: pd.DataFrame
    datalog: Dict[Any, Any]
    name: str


def get_timestamp():
    return datetime.datetime.now().strftime("%m%d%H%M%S%s")


def write_csvfile(data: pd.DataFrame, filepath: str, filename: str):
    complete_path = Path(f"{DATA_FOLDER}/{filepath}/{filename}")
    complete_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(complete_path, index=False)


def write_csvfiles(
    list_of_dataframes: List[pd.DataFrame], list_of_filenames: List[str]
):
    assert len(list_of_dataframes) == len(list_of_filenames)
    filepath = get_timestamp()
    for df, name in zip(list_of_dataframes, list_of_filenames):
        write_csvfile(data=df, filepath=filepath, filename=name)


def make_dgp_log(domain: Domain, sample: Sample):
    domain_dict = asdict(domain)
    sample_dict = asdict(sample)
    combined_dict = {"domain": domain_dict, "sample": sample_dict}
    return combined_dict


def write_dict_to_json(data: Dict[Any, Any], filepath: str, filename: str):
    complete_path = Path(f"{DATA_FOLDER}/{filepath}/{filename}")
    complete_path.parent.mkdir(parents=True, exist_ok=True)

    with open(complete_path, "w") as json_file:
        json.dump(data, json_file, indent=4, cls=NumpyEncoder)


def make_dataset(domain: Domain, sample: Sample, name: str):
    data = SampleGenerator.generate_data(domain=domain, sample=sample)
    datalog = make_dgp_log(domain=domain, sample=sample)
    return DataSet(data=data, datalog=datalog, name=name)


def materialize_dataset(dataset: DataSet, filepath: str):
    write_csvfile(data=dataset.data, filepath=filepath, filename=f"{dataset.name}.csv")
    write_dict_to_json(
        data=dataset.datalog, filepath=filepath, filename=f"{dataset.name}.json"
    )


def materialize_datasets(list_of_datasets: List[DataSet]):
    timestamp = get_timestamp()
    for ds in list_of_datasets:
        materialize_dataset(dataset=ds, filepath=timestamp)


dom1 = Domain(
    10,
    10,
    11,
    1354,
    104,
    10,
    10,
    0,
    0.5,
    3,
)


sample1 = Sample(1000, 1523213)
sample2 = Sample(1000, 25345)

dataset1 = make_dataset(dom1, sample1, "test")
dataset2 = make_dataset(dom1, sample2, "train")

lod = [dataset1, dataset2]

materialize_datasets(lod)
