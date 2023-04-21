from pathlib import Path

from src.util.config_reader import Configuration
from pandas import DataFrame, read_csv
import shutil
 



def write_data(data: DataFrame, filepath:str):
    c = Configuration().get()
    data_root_folder = c['DATA_HANDLING']["data_root_folder"]
    complete_path = Path(f'{data_root_folder}/{filepath}')  
    complete_path.parent.mkdir(parents=True, exist_ok=True)  
    data.to_csv(complete_path, index=False)


def read_data(filepath:str):
    c = Configuration().get()
    data_root_folder = c['DATA_HANDLING']["data_root_folder"]
    comlete_path = f"{data_root_folder}/{filepath}"
    data =read_csv(comlete_path)
    return data


def reset_data_storage():
    #delete
    c = Configuration().get()
    data_root_folder = c['DATA_HANDLING']["data_root_folder"]

    confirmation = ""
    while not ["y","n"].__contains__(confirmation):
        confirmation = input(f"Delete everything in folder /{data_root_folder}/? [y/n]")
    
    if confirmation == "y":
        try:
            shutil.rmtree(data_root_folder)
        except:
            print("Data root folder not found.")
        #recreate
        Path(data_root_folder).mkdir(parents=True, exist_ok=True)
        print(f"Reset data root folder complete: /{data_root_folder}/")
    else:
        print(f"Did not reset data root folder: /{data_root_folder}/")
