import os

import src.xgboost_training.strategies as ts

strategies = [
    ts.TgtOnlyStrategy,
    ts.SrcOnlyStrategy,
    ts.CombinationStrategy,
    ts.ContinuationStrategy,
    ts.ContinuationStrategyES,
    ts.RevisionStrategy,
]

files = ["tgtonly_result.json",
         "srconly_result.json",
         "combination_result.json",
         "continuation_result.json",
         "continuation_es_result.json"
         "revision_result.json"]

directory = "data/simulation"
scenario_paths = [f"{directory}/{s}" for s in os.listdir(directory)]
for sp in scenario_paths:
    experiment_paths = [f"{sp}/{e}" for e in sorted(os.listdir(sp))]
    for ep in experiment_paths:
        for Strategy, file in zip(strategies, files):
            #if not file in os.listdir(ep):
            print(ep, str(Strategy))
            st = Strategy(ep)
            st.execute()

    


