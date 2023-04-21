from dataclasses import dataclass, field
from multiprocessing import Pool
from typing import Callable

from src.util.logger import log_info


@dataclass
class Task():
    """
    Task
    """
    to_do_function: Callable
    to_do_list: list = field(default_factory=list)
     
class TaskExecutor():
    """
    TaskExecutor
    """
    def __init__(self) -> None:
        pass
     
    def run(self, task:Task, processes: int = 2):
        """ Executes a Task

         Args:
             task (Task): a Task object
             processes (int, optional): How many precesses should run. Defaults to 2.

         Returns:
             _type_: _description_
        """
        if __name__ == '__main__':
            log_info(f'Starting parallel processing with {processes} processes for {task}')
            with Pool(processes) as p:
                 result = p.map(task.to_do_function, task.to_do_list)
            log_info('Finished parallel processing')
            return result

