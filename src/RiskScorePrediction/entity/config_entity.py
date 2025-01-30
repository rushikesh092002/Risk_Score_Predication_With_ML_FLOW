from dataclasses import dataclass
from pathlib import Path


@dataclass
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_dir: Path
    train_data_dir: Path
    test_data_dir: Path    



@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    input_data_dir: Path
    STATUS_FILE: str
    all_schema: dict   

@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    train_data_dir: Path
    test_data_dir: Path
    transformed_train_data_dir: Path
    transformed_test_data_dir: Path

@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    train_data_dir: Path
    test_data_dir: Path
    model_name: str
    iterations: int
    learning_rate: float 
    depth: int               
    loss_function: str   
    early_stopping_rounds: int