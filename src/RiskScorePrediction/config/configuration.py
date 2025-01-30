from RiskScorePrediction.constants import *
from RiskScorePrediction.utils.common import read_yaml,create_directories
from RiskScorePrediction.entity.config_entity import DataIngestionConfig,DataValidationConfig,DataTransformationConfig

class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH,
        schema_filepath = SCHEMA_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.schema = read_yaml(schema_filepath)

        create_directories([self.config.artifacts_root])
    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])
        data_ingestion_config = DataIngestionConfig(
            root_dir = config.root_dir,
            source_URL = config.source_URL,
            local_data_dir = config.local_data_dir,
            train_data_dir=config.train_data_dir,
            test_data_dir=config.test_data_dir
        )
        return data_ingestion_config
    
    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation
        schema = self.schema.COLUMNS
        

        create_directories([config.root_dir])

        data_validation_config = DataValidationConfig(
            root_dir = config.root_dir,
            input_data_dir = config.input_data_dir,
            STATUS_FILE = config.STATUS_FILE,
            all_schema= schema
        )
        return data_validation_config
    
    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation

        create_directories([config.root_dir])
        data_transformation_config = DataTransformationConfig(
            root_dir = config.root_dir,
            train_data_dir = config.train_data_dir,
            test_data_dir = config.test_data_dir,
            transformed_train_data_dir = config.transformed_train_data_dir,
            transformed_test_data_dir = config.transformed_test_data_dir
        )

        return data_transformation_config