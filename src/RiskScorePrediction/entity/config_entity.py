from dataclasses import dataclass
from pathlib import Path

@dataclass
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_dir: Path
    train_data_dir: Path
    test_data_dir: Path    