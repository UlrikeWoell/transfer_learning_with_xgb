import csv
import json
from typing import Any, Dict, Union
from dataclasses import asdict, is_dataclass
import numpy as np
from xgboost import XGBClassifier, Booster

def flatten_dict(data: Dict[str, Union[Dict[str, Any], Any]], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
    flattened_dict: Dict[str, Any] = {}
    for key, value in data.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key
        if isinstance(value, dict):
            flattened_dict.update(flatten_dict(value, new_key, sep=sep))
        else:
            flattened_dict[new_key] = value
    return flattened_dict

def write_dict_to_csv(data: Dict[str, Union[Dict[str, Any], Any]], file_path: str) -> None:
    flattened_data = flatten_dict(data)
    keys = flattened_data.keys()
    with open(file_path, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=keys)
        if csvfile.tell() == 0:
            writer.writeheader()
        writer.writerow(flattened_data)


def write_dataclass_to_csv(data: object, file_path:str)-> None:
    assert is_dataclass(data)
    datadict = asdict(data)
    write_dict_to_csv(datadict,file_path=file_path)


# Custom JSON Encoder with support for ndarray
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()  
        if isinstance(obj, XGBClassifier):
            return obj.__dict__
        if isinstance(obj, Booster):
            return ""
        return super().default(obj)


def write_dataclass_to_json(data: object, file_path:str)-> None:
    assert is_dataclass(data)
    datadict = asdict(data)
    with open(file_path, 'a') as json_file:
        json.dump(datadict, json_file, cls=CustomJSONEncoder)