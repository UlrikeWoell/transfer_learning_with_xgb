import csv
from typing import Dict, Any

class ResultAggregator:
    def __init__(self, output_file:str):
        self.output_file = output_file
        self.results = []

    def add_result(self, folder_name:str, tuning_results:Dict['str',Any]):
        result = {'folder_name': folder_name}
        result.update(tuning_results)
        self.results.append(result)

    def save_results(self):
        with open(self.output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.results[0].keys())
            writer.writeheader()
            writer.writerows(self.results)
