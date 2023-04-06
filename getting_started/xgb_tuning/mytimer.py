"""
A simple timer to record start time, end time and duration
"""

from datetime import datetime


class Timer():

    """
    A simple timer to record start time, end time and duration
    """

    def __init__(self) -> None:
        self.start_time = None
        self.end_time = None
        self.duration = None
        self.collection = None

    def start(self):
        """Sets now as start time
        """
        self.start_time = datetime.now()

    def end(self):
        """Sets now as end time"""
        self.end_time = datetime.now()
        self.duration = self.end_time - self.start_time

    def print(self, verbose=False):
        """Prints recorded duration

        Args:
            verbose (bool, optional): Prints start and and end if True. Defaults to False.
        """
        if verbose:
            print(f'Start: {self.start_time}')
            print(f'End: {self.end_time}')
        print(f'Duration: {self.duration}')
