from dataclasses import dataclass, field
from pprint import pprint
import time
from src.domains.domains import Domain, DomainChanger
from src.transfer_strategies.transfer_strategies import (
    SRCONLY,
    TGTONLY,
    SamplingTask,
    SimulationSetUp,
)
from src.util.dataclass_to_file import write_dataclass_to_csv, write_dataclass_to_json


@dataclass
class Run:
    domain_seed: int
    n_features: int


runs = [Run(s, s + 1) for s in range(10, 12)]


def simulate(run: Run):
    ds = DomainChanger()
    simulator = SRCONLY()
    src_dom = Domain(run.domain_seed, run.n_features)
    tgt_dom = Domain(run.domain_seed, run.n_features)

    aucs = dict()

    for n in [
        100,
        200,
        400,
        # 801, #1_600, #3_201,
        # 6_400, 12_800, 25_600, 51_200
    ]:
        src_sampling = SamplingTask(n=n, sample_seed=n)
        tgt_sampling = SamplingTask(n=n, sample_seed=n + 1)

        setup = SimulationSetUp(
            source_domain=src_dom,
            target_domain=tgt_dom,
            sampling_task_source_train=src_sampling,
            sampling_task_target_train=tgt_sampling,
        )

        sim_result = simulator.execute_strategy(setup)
        write_dataclass_to_json(sim_result, "data/sim.json")

start = time.time()
for r in runs:
    simulate(r)

elapsed = time.time() - start
print(f'elapased: {elapsed}')