import os

import src.xgboost_training.strategies as ts

strategies = [
    ts.TgtOnlyStrategy,
    ts.SrcOnlyStrategy,
    ts.CombinationStrategy,
    ts.FreezingStrategy,
    ts.ProgressiveLearningStrategy,
]

files = [
    "tgtonly_result.json",
    "srconly_result.json",
    "combination_result.json",
    "freeze_result.json",
    "progressive_result.json",
]

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
    experiment_paths = [f"{sp}/{e}" for e in sorted(os.listdir(sp))]
    for ep in experiment_paths:
        for Strategy, file in zip(strategies, files):            
            if not file in os.listdir(ep):
            #if True:
                print(ep, str(Strategy))
                st = Strategy(ep)
                st.execute()
        
        for ro in finetuning_options:
            augment=ro[0]
            prune=ro[1]
            reweigth=ro[2]
            filename = f"finetuning_result_{'a' if augment else ''}{'p' if prune else ''}{'r' if reweigth else ''}.json"

            if not filename in os.listdir(ep):
                print(ep, filename)
                st = ts.FinetuningStrategy(path=ep, augment=augment, prune=prune, reweight=reweigth)
                st.execute()
        
print('Build models: Done')