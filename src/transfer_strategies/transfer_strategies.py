from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from src.domains.domains import Domain
from src.transfer_strategies.transfer_strategies import SimulationSetUp, SimulationSummary, SimulationSummary_SRCONLY
from src.xgb_tuning.xgb_base_training import tune,  get_model_evaluation, TuningOutput, ModelEvaluation
import pandas as pd
from typing import Tuple, Any

@dataclass
class SamplingTask():
    domain: Domain
    n: int
    sample_seed: int


@dataclass
class SimulationSetUp(ABC):
    sampling_task_source_train : SamplingTask 
    sampling_task_target_train: SamplingTask
    sampling_task_source_test : SamplingTask =field(init=False)
    sampling_task_target_test: SamplingTask=field(init=False)

    def __post_init__(self):
        self.sampling_task_source_test = self.get_test_task(self.sampling_task_source_train)
        self.sampling_task_target_test = self.get_test_task(self.sampling_task_target_train)

    def get_test_task(self, train_task: SamplingTask):
        return SamplingTask(domain= train_task.domain,
                            n = 1000,
                            sample_seed=train_task.sample_seed+1)

@dataclass
class SimulationSummary(ABC):
    ...

@dataclass
class SimulationSummary_SRCONLY(SimulationSummary):
    simulation_set_up: SimulationSetUp
    model_evaluation_on_src: ModelEvaluation
    model_evaluation_on_tgt: ModelEvaluation


class AbstractTransferStrategy(ABC):
    """
    Abstract transfer strategy
    """
    
    def sample_from_source_train(self, rec: SimulationSetUp)-> Tuple[pd.DataFrame, pd.DataFrame]:
        return self.sample(rec.sampling_task_source_train)
    
    def sample_from_source_test(self, rec: SimulationSetUp)-> Tuple[pd.DataFrame, pd.DataFrame]:
        return self.sample(rec.sampling_task_source_test)
    
    def sample_from_target_train(self, rec: SimulationSetUp)-> Tuple[pd.DataFrame, pd.DataFrame]:
        return self.sample(rec.sampling_task_target_train)
    
    def sample_from_target_test(self, rec: SimulationSetUp)-> Tuple[pd.DataFrame, pd.DataFrame]:
        return self.sample(rec.sampling_task_target_test)

    def sample(self, sampling_task: SamplingTask) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return sampling_task.domain.generate_data(n=sampling_task.n,
                                                   sample_seed=sampling_task.sample_seed)
    
    @abstractmethod
    def get_summary(self,
                    model_eval_on_src: ModelEvaluation|None,
                    model_eval_on_tgt: ModelEvaluation|None,
                    simulation_setup: SimulationSetUp) -> SimulationSummary:
        ...

    @abstractmethod
    def execute_simulation(self, simulation_setup: SimulationSetUp) -> SimulationSummary:
        ...

 
class SRCONLY(AbstractTransferStrategy):
    "trains a model on the source data only. Evaluates the source model on the target data"

    def get_summary(self,
                    model_eval_on_src: ModelEvaluation,
                    model_eval_on_tgt: ModelEvaluation,
                    simulation_setup: SimulationSetUp) -> SimulationSummary_SRCONLY:
        return SimulationSummary_SRCONLY(model_evaluation_on_src=model_eval_on_src, 
                                 model_evaluation_on_tgt=model_eval_on_tgt,
                                 simulation_set_up=simulation_setup)

    def execute_simulation(self, simulation_setup: SimulationSetUp) -> SimulationSummary:
        src_X_train, src_y_train = self.sample_from_source_train(simulation_setup)
        tuning_result = tune(src_X_train, y_train=src_y_train)

        src_X_test, src_y_test = self.sample_from_source_test(simulation_setup)
        src_evaluation = get_model_evaluation(tuning_output=tuning_result, X_test = src_X_test, y_test = src_y_test)

        tgt_X_test, tgt_y_test = self.sample_from_target_test(simulation_setup)
        tgt_evaluation = get_model_evaluation(tuning_output=tuning_result, X_test=tgt_X_test, y_test=tgt_y_test)

        summary = self.get_summary(src_evaluation, tgt_evaluation, simulation_setup)

        return summary


@dataclass 
class SimulationSummary_TGTONLY(SimulationSummary):
    model_eval_on_tgt: ModelEvaluation | None
    simulation_setup: SimulationSetUp | None


class TGTONLY(AbstractTransferStrategy):
    """trains a model on the source domain only. Evaluates the target model on the target data
    """
    def get_summary(self, 
                    model_eval_on_src: ModelEvaluation|None, 
                    model_eval_on_tgt: ModelEvaluation|None, 
                    simulation_setup: SimulationSetUp) -> SimulationSummary:
        return SimulationSummary_TGTONLY(model_eval_on_tgt=model_eval_on_tgt, 
                                         simulation_setup=simulation_setup)

    def execute_simulation(self, simulation_setup: SimulationSetUp) -> SimulationSummary:

        tgt_X_train, tgt_y_train = self.sample_from_target_train(simulation_setup)
        tuning_result = tune(tgt_X_train, y_train=tgt_y_train)
        
        tgt_X_test, tgt_y_test = self.sample_from_target_test(simulation_setup)
        tgt_evaluation = get_model_evaluation(tuning_output=tuning_result, X_test=tgt_X_test, y_test=tgt_y_test)

        summary = self.get_summary(model_eval_on_src=None,
                                   model_eval_on_tgt=tgt_evaluation,
                                   simulation_setup= simulation_setup)

        return summary

@dataclass 
class SimulationSummary_WEIGHTED(SimulationSummary):
    simulation_setup: SimulationSetUp
    model_evaluation_on_tgt: ModelEvaluation


class WEIGHTED(AbstractTransferStrategy):
    def merge_datasets(self, 
                       src_X:pd.DataFrame, 
                       src_y:pd.DataFrame, 
                       tgt_X:pd.DataFrame, 
                       tgt_y: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        merged_X = pd.concat(src_X, tgt_X)
        merged_y = pd.concat(src_y, tgt_y)
        return merged_X, merged_y
    
    def get_weight_vector(self, src_y: pd.DataFrame, tgt_y:pd.DataFrame):
        src_n = src_y.shape[1]
        tgt_n = tgt_y.shape[1]
        src_weights = [tgt_n/src_n for _ in range(src_n)]
        tgt_weights = [1.0 for _ in range(tgt_n)]
        all_weights = src_weights.append(tgt_weights)
        return all_weights

    def get_summary(self, 
                    model_eval_on_src: ModelEvaluation | None, 
                    model_eval_on_tgt: ModelEvaluation | None, 
                    simulation_setup: SimulationSetUp) -> SimulationSummary:
        return SimulationSummary_WEIGHTED(simulation_setup = simulation_setup,
                                          model_evaluation_on_tgt=model_eval_on_tgt)
    

    def execute_simulation(self, simulation_setup: SimulationSetUp) -> SimulationSummary:
        src_X_train, src_y_train = self.sample_from_source_train(simulation_setup)
        tgt_X_train, tgt_y_train = self.sample_from_target_train(simulation_setup)
        merged_X_train, merged_y_train = self.merge_datasets(src_X_train, src_y_train, tgt_X_train, tgt_y_train)
        weights = self.get_weight_vector(src_y_train, tgt_y_train)
        
        tuning = tune(X_train=merged_X_train, y_train=merged_y_train, sample_weights = weights)
        
        tgt_X_test, tgt_y_test = self.sample_from_target_test(simulation_setup)
        tgt_evaluation = get_model_evaluation(tuning_output=tuning, X_test=tgt_X_test, y_test=tgt_y_test)

        summary = self.get_summary(model_eval_on_src=None,
                                   model_eval_on_tgt=tgt_evaluation,
                                   simulation_setup=simulation_setup)
        return summary


class REVISE(AbstractTransferStrategy):


class CONTINUE(AbstractTransferStrategy):
    

