from src.domains.domains import Domain, DomainChanger
from src.transfer_strategies.transfer_strategies import AbstractTransferStrategy, SamplingTask, SimulationSetUp
from src.util.dataclass_to_file import write_dataclass_to_json
from time import time
from dataclasses import dataclass, field, asdict

from src.util.config_reader import Configuration



@dataclass
class RunResult():
    start_time_ns: float 
    end_time_ns: float

    strategy: str
    domain_seed: int
    n_features: int

    src_train_n: int
    tgt_train_n:int
    src_test_n:int
    tgt_test_n: int

    src_train_seed: int
    tgt_train_seed:int
    src_test_seed: int
    tgt_test_seed:int
    
    src_domain: dict = field(default_factory=dict) # use asdict(src_dom)
    tgt_domain: dict = field(default_factory=dict) # use asdict(tgt_dom)
    domain_changes: list = field(default_factory=list)

    src_tgt_auc_diff: float
    auc: float

    def to_json(self):
        write_dataclass_to_json(self)

    



def simulate(run: Run) -> RunResult:
    start_time = time()
    ds = DomainChanger()
    src_dom = Domain(run.domain_seed, run.n_features)
    tgt_dom = Domain(run.domain_seed, run.n_features)

    setup = SimulationSetUp(
        source_domain=src_dom,
        target_domain=tgt_dom,
        sampling_task_source_train=run.src_sampling_task,
        sampling_task_target_train=run.tgt_sampling_task,
    )

    sim_result = run.strategy.execute_strategy(setup)
    end_time = time()
    write_dataclass_to_json(sim_result, "data/sim.json")