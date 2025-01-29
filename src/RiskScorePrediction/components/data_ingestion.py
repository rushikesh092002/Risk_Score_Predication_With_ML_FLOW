import os
import urllib.request as request
import zipfile
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from RiskScorePrediction import logger
from RiskScorePrediction.entity.config_entity import DataIngestionConfig
from RiskScorePrediction.utils.common import get_size




class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config
    

    def download_file(self):
        if not os.path.exists(self.config.local_data_dir):
            filename, headers = request.urlretrieve(
                url = self.config.source_URL,
                filename= self.config.local_data_dir
            )
        else:
            logger.info(f"Downloaded {get_size(Path(self.config.local_data_dir))} bytes")

    def train_test_split(self):
        try:
            df= pd.read_csv("artifacts/data_ingestion/data.csv")
            train_set,test_set = train_test_split(df,test_size=0.2,random_state=42)
            
            train_set.to_csv(self.config.train_data_dir,index=False,header=True)

            test_set.to_csv(self.config.test_data_dir,index=False,header=True)

            logger.info("Train-Test Split Done")
        except Exception as e:
            raise e








