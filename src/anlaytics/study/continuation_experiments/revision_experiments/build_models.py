import os

from src.xgboost_training.strategies import ContinuationStrategyES

directory = "data/simulation"
scenario_paths = [f"{directory}/{s}" for s in os.listdir(directory)]
for sp in scenario_paths:
    experiment_paths = [f"{sp}/{e}" for e in os.listdir(sp)]
    for ep in experiment_paths:
        print(ep)
        st = ContinuationStrategyES(path=ep)
        st.execute()
