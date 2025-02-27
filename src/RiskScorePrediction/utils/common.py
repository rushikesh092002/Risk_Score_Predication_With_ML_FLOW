import os
from box.exceptions import BoxValueError
import yaml
from RiskScorePrediction import logger
import json
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any
import dill
import pickle
import joblib




@ensure_annotations
def read_yaml(path_to_yaml:Path) -> ConfigBox:
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file : {path_to_yaml} loaded  sucessfully")
            return ConfigBox(content)
        
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except BoxValueError as e:
        raise e
    


@ensure_annotations
def create_directories(path_to_directories:list, verbose=True):
    for path in path_to_directories:
        os.makedirs(path , exist_ok=True)
        if verbose:
            logger.info(f"created directory at :{path}")


@ensure_annotations
def save_json(path: Path , data: dict):
    with open(path, "w") as f:
        json.dump(data, f , indent=4)
    
    logger.info(f"json file saved at :{path}")


@ensure_annotations
def get_size(path: Path) -> str:
    size_in_kb = round(os.path.getsize(path)/1024)
    return f"~ {size_in_kb} KB"

def save_object(file_path,obj):
    dir_path=os.path.dirname(file_path)

    os.makedirs(dir_path,exist_ok=True)

    with open(file_path,"wb") as file_obj:
        pickle.dump(obj,file_obj)



def load_object(file_path):
    try:
        
        if file_path.endswith(".pkl"):
            with open(file_path, 'rb') as file:
                return pickle.load(file)
        elif file_path.endswith(".joblib"):
            return joblib.load(file_path)
        else:
            raise ValueError("Unsupported file extension. Use .pkl or .joblib.")
    except Exception as e:
        raise e
    