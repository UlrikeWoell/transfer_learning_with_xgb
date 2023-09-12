import json
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

import pandas as pd

from src.xgboost_training.data_loader import TransferDataLoader
from src.xgboost_training.xgboost_transfer_tuner import XGBoostTransferTuner
from src.xgboost_training.xgboost_tuner_config import DefaultXGBoostTunerConfig


class StrategyExecutor(ABC):
    def __init__(self, path: str) -> None:
        self.path = path

    @abstractmethod
    def execute(self) -> None:
        """Executes the stratgy on the given files in the path.
        Creates a {}_result.json file in the path"""
        ...

    @abstractmethod
    def result_path(self) -> str:
        """
        Complete path where the result will be created, specifying the file name

        Returns:
            str: f'{self.path}/{_}_result.json'
        """

    def load_data(self):
        loader = TransferDataLoader(self.path)
        return loader.load_data()

    def setup_tuner(self):
        tuner_config = DefaultXGBoostTunerConfig
        tuner = XGBoostTransferTuner(tuner_config)
        return tuner

    def save_results_to_json(self, tuning_results: Dict[str, Any]):
        result_file = self.result_path()
        with open(result_file, "w") as f:
            json.dump(tuning_results, f)

    def save_failed_result_to_json(self, msg: str | None = None):
        tuning_result = {"success": False, "msg": msg}
        self.save_results_to_json(tuning_results=tuning_result)


class TgtOnlyStrategyExecutor(StrategyExecutor):
    def execute(self) -> None:
        tuner = self.setup_tuner()
        # try:
        data = self.load_data()
        tuning_results, _ = tuner.tune_model(
            X_train=data["tgt_train_X"],
            y_train=data["tgt_train_y"],
            X_test=data["tgt_test_X"],
            y_test=data["tgt_test_y"],
        )
        self.save_results_to_json(tuning_results)

    def result_path(self) -> str:
        return f"{self.path}/tgtonly_result.json"


class SrcOnlyStrategyExecutor(StrategyExecutor):
    def execute(self) -> None:
        tuner = self.setup_tuner()
        try:
            data = self.load_data()
            tuning_results, _ = tuner.tune_model(
                X_train=data["src_train_X"],
                y_train=data["src_train_y"],
                X_test=data["tgt_test_X"],
                y_test=data["tgt_test_y"],
            )
            self.save_results_to_json(tuning_results)

        except Exception as ex:
            print(f"Error processing folder {self.path}:\n {str(ex)}")
            self.save_failed_result_to_json(
                msg=str(ex),
            )

    def result_path(self) -> str:
        return f"{self.path}/srconly_result.json"


class CombinationStrategyExecutor(StrategyExecutor):
    def execute(self) -> None:
        tuner = self.setup_tuner()
        try:
            data = self.load_data()
            tuning_results = tuner.tune_combination_model(data)
            self.save_results_to_json(tuning_results)

        except Exception as ex:
            print(f"Error processing folder {self.path}:\n {str(ex)}")
            self.save_failed_result_to_json(
                msg=str(ex),
            )

    def result_path(self) -> str:
        return f"{self.path}/combination_result.json"


class FreezingStrategy(StrategyExecutor):
    def execute(self) -> None:
        tuner = self.setup_tuner()
        try:
            data = self.load_data()
            tuning_results = tuner.tune_freeze_model(
                X_train_1=data["src_train_X"],
                y_train_1=data["src_train_y"],
                X_train_2=data["tgt_train_X"],
                y_train_2=data["tgt_train_y"],
                X_test=data["tgt_test_X"],
                y_test=data["tgt_test_y"],
            )
            self.save_results_to_json(tuning_results)

        except Exception as ex:
            print(f"Error processing folder {self.path}:\n {str(ex)}")
            self.save_failed_result_to_json(
                msg=str(ex),
            )

    def result_path(self) -> str:
        return f"{self.path}/freeze_result.json"


class ProgressiveLearningStrategy(StrategyExecutor):
    def execute(self) -> None:
        tuner = self.setup_tuner()
        try:
            data = self.load_data()
            tuning_results = tuner.tune_progressive_model(
                X_train_1=data["src_train_X"],
                y_train_1=data["src_train_y"],
                X_train_2=data["tgt_train_X"],
                y_train_2=data["tgt_train_y"],
                X_test=data["tgt_test_X"],
                y_test=data["tgt_test_y"],
            )
            self.save_results_to_json(tuning_results)
            print(self.path)

        except Exception as ex:
            print(f"Error processing folder {self.path}:\n {str(ex)}")
            self.save_failed_result_to_json(
                msg=str(ex),
            )

    def result_path(self) -> str:
        return f"{self.path}/progressive_result.json"


class FinetuningStrategy(StrategyExecutor):
    def __init__(
        self, path: str, augment: bool = True, prune: bool = True, reweight: bool = True
    ) -> None:
        self.path = path
        self.augment = augment
        self.prune = prune
        self.reweight = reweight

    def save_results_to_json(
        self, tuning_results: Dict[str, Any], augment: bool, prune: bool, reweight: bool
    ):
        result_file = self.result_path(augment, prune, reweight)
        with open(result_file, "w") as f:
            json.dump(tuning_results, f)

    def save_failed_result_to_json(
        self, msg: str | None, augment: bool, prune: bool, reweight: bool
    ):
        tuning_result = {"success": False, "msg": msg}
        self.save_results_to_json(tuning_result, augment, prune, reweight)

    def execute(self) -> None:
        tuner = self.setup_tuner()
        try:
            data = self.load_data()
            tuning_results = tuner.tune_finetuned_model(
                X_train_src=data["src_train_X"],
                y_train_src=data["src_train_y"],
                X_train_tgt=data["tgt_train_X"],
                y_train_tgt=data["tgt_train_y"],
                X_test_src=data["src_test_X"],
                y_test_src=data["src_test_y"],
                X_test_tgt=data["tgt_test_X"],
                y_test_tgt=data["tgt_test_y"],
                augment=self.augment,
                prune=self.prune,
                reweight=self.reweight,
            )
            self.save_results_to_json(
                tuning_results,
                augment=self.augment,
                prune=self.prune,
                reweight=self.reweight,
            )

        except Exception as ex:
            print(f"Error processing folder {self.path}:\n {str(ex)}")
            self.save_failed_result_to_json(
                msg=str(ex),
                augment=self.augment,
                prune=self.prune,
                reweight=self.reweight,
            )
            raise ex

    def result_path(self, augment: bool, prune: bool, reweight: bool) -> str:
        filename = f"finetuning_result_{'a' if augment else ''}{'p' if prune else ''}{'r' if reweight else ''}.json"
        return f"{self.path}/{filename}"
