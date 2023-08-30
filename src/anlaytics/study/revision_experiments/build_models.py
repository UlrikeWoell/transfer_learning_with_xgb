import os

from src.xgboost_training.strategies import FinetuningStrategy

finetuning_options = [
    [False, False, False],
    [False, False, True],
    [False, True, False],
    [False, True, True],
    [True, False, False],
    [True, False, True],
    [True, True, False],
    [True, True, True],
]

directory = "data/simulation"
scenario_paths = [f"{directory}/{s}" for s in os.listdir(directory)]
for sp in scenario_paths:
    experiment_paths = [f"{sp}/{e}" for e in os.listdir(sp)]
    for ep in experiment_paths:
        for ro in finetuning_options:
            augment=ro[0]
            prune=ro[1]
            reweight=ro[2]
            filename = f"finetuning_result_{"a" if augment else ''}{"p" if prune else ''}{"r" if reweight else ''}.json"
            print(ep, filename)
            st = FinetuningStrategy(path=ep, augment=ro[0], prune=ro[1], reweight=ro[2])
            st.execute()
