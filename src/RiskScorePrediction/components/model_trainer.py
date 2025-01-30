import pandas as pd
import os
from RiskScorePrediction import logger
from catboost import CatBoostRegressor
import joblib
from RiskScorePrediction.entity.config_entity import ModelTrainerConfig

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
        try:
            
            train_data = pd.read_csv(self.config.train_data_dir)
            test_data = pd.read_csv(self.config.test_data_dir)

           
            train_data.columns = train_data.columns.astype(str).str.strip()
            test_data.columns = test_data.columns.astype(str).str.strip()

            
            target_col = 'RiskScore'  

           
            print(f"Target Column: {target_col}")
            print(f"Available Columns: {train_data.columns.tolist()}")

            
            if target_col not in train_data.columns:
                raise KeyError(f"Target column '{target_col}' not found! Available columns: {train_data.columns.tolist()}")

          
            train_x = train_data.drop(columns=[target_col])
            test_x = test_data.drop(columns=[target_col])
            train_y = train_data[target_col]
            test_y = test_data[target_col]

            
            if train_y.isna().sum() > 0:
                print(f"Warning: There are {train_y.isna().sum()} NaN values in the target column. Filling with mean.")
                train_y.fillna(train_y.mean(), inplace=True)  # Fill NaN values with the mean of the column

            if test_y.isna().sum() > 0:
                print(f"Warning: There are {test_y.isna().sum()} NaN values in the target column. Filling with mean.")
                test_y.fillna(test_y.mean(), inplace=True)  # Fill NaN values with the mean of the column

            
            assert not train_x.empty, "train_x is empty after dropping the target column!"
            assert not train_y.empty, "train_y is empty!"

          
            cat = CatBoostRegressor(
                iterations=self.config.iterations,
                learning_rate=self.config.learning_rate,
                depth=self.config.depth,
                loss_function=self.config.loss_function,
                early_stopping_rounds=self.config.early_stopping_rounds  
            )

            
            cat.fit(train_x, train_y, eval_set=(test_x, test_y))

           
            joblib.dump(cat, os.path.join(self.config.root_dir, self.config.model_name))

        except Exception as e:
            raise e
