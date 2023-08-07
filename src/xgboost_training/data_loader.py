from typing import Any, Dict

import pandas as pd

TARGET_ID = "y"

class DataLoader:
    def __init__(self, file_path: str, train_file_name: str, test_file_name: str):

        self.train_filepath = f'{file_path}/{train_file_name}'
        self.test_filepath = f'{file_path}/{test_file_name}'

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
