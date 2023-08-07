from src.xgboost_training.tuning_routine import (
    TuningRoutine,
    TuningRoutineConfiguration,
)
import datetime

def run():
    config = TuningRoutineConfiguration(
        data_dir="data/validate_dgp",
        output_file="combined_results.csv",
        test_file_name="test.csv",
        train_file_name="train.csv",
        tuning_result_file_name="tuning_results.json",
    )

    routine = TuningRoutine(config)
    print(f'Start time: {datetime.datetime.now()}')
    routine.start()
    print(f'End time: {datetime.datetime.now()}')

run()