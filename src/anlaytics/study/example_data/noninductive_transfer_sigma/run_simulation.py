from src.xgboost_training.tuning_routine import (
    TuningRoutine,
    TuningRoutineConfiguration,
)
import datetime

def run_SRC_ONLY():
    config = TuningRoutineConfiguration(
        data_dir="data/noninductive_transfer/sigma",
        output_file="combined_results.csv",
        test_file_name="t_test.csv",
        train_file_name="s_train.csv",
        tuning_result_file_name="SRCONLY_tuning_results.json",
    )

    routine = TuningRoutine(config)
    print(f'Start time: {datetime.datetime.now()}')
    routine.start()
    print(f'End time: {datetime.datetime.now()}')


def run_TGT_ONLY():
    config = TuningRoutineConfiguration(
    data_dir="data/noninductive_transfer/sigma",
    output_file="combined_results.csv",
    test_file_name="t_test.csv",
    train_file_name="t_train.csv",
    tuning_result_file_name="TGTONLY_tuning_results.json",
)

    routine = TuningRoutine(config)
    print(f'Start time: {datetime.datetime.now()}')
    routine.start()
    print(f'End time: {datetime.datetime.now()}')

run_TGT_ONLY()