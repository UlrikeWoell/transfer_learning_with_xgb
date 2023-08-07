from src.domains.domains import Domain, DomainChanger
from src.transfer_strategies.transfer_strategies import (
    SRCONLY,
    TGTONLY,
    SamplingTask,
    SimulationSetUp,
)
from pprint import pprint

ooo = DomainChanger()


tgt_only = TGTONLY()
src_only = SRCONLY()



for f in [0, 0.1, 0.2, 1, 1.4]:
    orig = Domain(domain_seed=1, base_feature_count=3)
    
    chan = Domain(domain_seed=1, base_feature_count=3)
    chan = ooo.change_bernoulli_bias(domain=orig, factor=f)

    sts = SamplingTask(100, 23742)
    stt = SamplingTask(100, 231432)
    simulation_setup = SimulationSetUp(
        source_domain=orig,
        target_domain=chan,
        sampling_task_source_train=sts,
        sampling_task_target_train=stt,
    )

    src_only_result = src_only.execute_strategy(simulation_setup)
    tgt_only_result = tgt_only.execute_strategy(simulation_setup)
    
    s_auc = src_only_result.model_evaluation_on_tgt.auc_score
    t_auc =tgt_only_result.model_evaluation_on_tgt.auc_score
    print(f'---{f}: s:{s_auc}, t:{t_auc}, s/t: {s_auc/t_auc}')

