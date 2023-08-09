import os
from dataclasses import dataclass, field

from src.xgboost_training.data_loader import DataLoader
from src.xgboost_training.xgboost_tuner import XGBoostTuner, XGBoostTunerConfig


@dataclass
class TuningRoutineConfiguration:
    data_dir: str
    output_file: str
    test_file_name: str
    train_file_name: str
    tuning_result_file_name: str


class TuningRoutine:
    def __init__(self, tuning_routine_config: TuningRoutineConfiguration) -> None:
        self.data_dir = tuning_routine_config.data_dir
        self.output_file = tuning_routine_config.output_file
        self.train_file = tuning_routine_config.train_file_name
        self.test_file = tuning_routine_config.test_file_name
        self.tuning_result_file_name = tuning_routine_config.tuning_result_file_name

    def start(self):
        """
        Tunes and tests XGB models given in data_dir.

        All train files must have identical names.
        All test files must have identical names.

        """
        tuner_config = XGBoostTunerConfig()

        for folder_name in sorted(os.listdir(self.data_dir)):
            print(folder_name)
            if os.path.isdir(os.path.join(self.data_dir, folder_name)):
                path = f"{self.data_dir}/{folder_name}"
                loader = DataLoader(
                    file_path=path,
                    test_file_name=self.test_file,
                    train_file_name=self.train_file,
                )
                tuner = XGBoostTuner(tuner_config)

                try:
                    data = loader.load_data()
                    tuning_results = tuner.tune_model(
                        X_train=data["X_train"],
                        y_train=data["y_train"],
                        X_test=data["X_test"],
                        y_test=data["y_test"],
                    )
                    tuner.save_results_to_json(
                        file_path=path,
                        file_name=self.tuning_result_file_name,
                        tuning_results=tuning_results,
                    )

                except Exception as ex:
                    print(f"Error processing folder {folder_name}: {str(ex)}")
                    tuner.save_failed_result_to_json(
                        file_path=path,
                        file_name=self.tuning_result_file_name,
                        msg=str(ex),
                    )
