from typing import Any, Dict, Literal

import pandas as pd

TARGET_ID = "y"


class SimpleDataLoader:
    """Loads 2 csv files for traditional ML problems:
    - train.csv, 
    - test.csv

    Returns X and y separately for all files.
    """

    def __init__(self, file_path: str, train_file_name: str, test_file_name: str):
        self.train_filepath = f"{file_path}/{train_file_name}"
        self.test_filepath = f"{file_path}/{test_file_name}"

    def load_data(self) -> Dict[str, Any]:
        train_data = pd.read_csv(self.train_filepath)
        test_data = pd.read_csv(self.test_filepath)

        X_train = train_data.drop(TARGET_ID, axis=1)
        y_train = train_data[TARGET_ID]

        X_test = test_data.drop(TARGET_ID, axis=1)
        y_test = test_data[TARGET_ID]

        return {
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test,
        }


class TransferDataLoader:
    """Loads 4 files for transfer learning:
    - src_train.csv
    - scr_trest.csv
    - tgt_train.csv
    - tgt_test.csv
    Returns X and y separately for all files

    """

    def __init__(
        self,
        file_path: str,
    ):
        self.path = file_path

    def file_name(
        self,
        domain_prefix: Literal["src", "tgt"],
        subset_suffix: Literal["train", "test"],
    ):
        return f"{domain_prefix}_{subset_suffix}.csv"

    def complete_path(self, filename: str):
        return f"{self.path}/{filename}"

    def load_data(self) -> Dict[str, pd.DataFrame]:
        prepared_data = {}
        for dom in ["src", "tgt"]:
            for subset in ["train", "test"]:
                filename = self.file_name(dom, subset)
                complete_path = self.complete_path(filename)
                data = pd.read_csv(complete_path)
                X = data.drop(TARGET_ID, axis=1)
                y = data[TARGET_ID]
                prepared_data[f"{dom}_{subset}_X"] = X
                prepared_data[f"{dom}_{subset}_y"] = y
                
        return prepared_data
