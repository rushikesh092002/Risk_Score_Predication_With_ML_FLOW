import os
from RiskScorePrediction import logger
import pandas as pd
from RiskScorePrediction.entity.config_entity import DataValidationConfig


class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config

    
    import pandas as pd

class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config

    def validate_all_columns(self) -> bool:
        try:
            validation_status = True  
            
            data = pd.read_csv(self.config.input_data_dir)
            all_cols = set(data.columns) 
            all_schema = set(self.config.all_schema.keys())  
            target_column = self.config.all_schema.get("TARGET_COLUMN")  
            if not all_schema.issubset(all_cols):
                validation_status = False
                missing_cols = all_schema - all_cols
                with open(self.config.STATUS_FILE, 'w') as f:
                    f.write(f"Validation Status: {validation_status}\nMissing Columns: {missing_cols}\n")

            if target_column and target_column not in all_cols:
                validation_status = False
                with open(self.config.STATUS_FILE, 'w') as f:
                    f.write(f"Validation Status: {validation_status}\nMissing Target Column: {target_column}\n")


            if validation_status:
                with open(self.config.STATUS_FILE, 'w') as f:
                    f.write(f"Validation Status: {validation_status}")

            return validation_status
        
        except Exception as e:
            raise e

