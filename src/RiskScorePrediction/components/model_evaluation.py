import os
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
import numpy as np
import joblib
from pathlib import Path
from RiskScorePrediction.utils.common import save_json
from RiskScorePrediction.entity.config_entity import ModelEvaluationConfig

import os
import shutil


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def eval_metrics(self, actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2

    def log_into_mlflow(self):
        
        os.environ['MLFLOW_TRACKING_URI'] = "https://dagshub.com/rushikesh092002/Risk_Score_Predication_With_ML_FLOW.mlflow"
        os.environ['MLFLOW_TRACKING_USERNAME'] = "rushikesh092002"
        os.environ['MLFLOW_TRACKING_PASSWORD'] = "fcbbf35c568823de8126089cd6a25a106bd8afc4"

        test_data = pd.read_csv(self.config.test_data_path)
        model = joblib.load(self.config.model_path)

        target_col = 'RiskScore'
        test_x = test_data.drop([target_col], axis=1)
        test_y = test_data[[target_col]]
        
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_artifact_uri()).scheme

        if mlflow.active_run():
            mlflow.end_run()

        with mlflow.start_run():
            predicted_qualities = model.predict(test_x)
            rmse, mae, r2 = self.eval_metrics(test_y, predicted_qualities)

            scores = {"rmse": rmse, "mae": mae, "r2": r2}
            save_json(path=Path(self.config.metric_file_name), data=scores)

            mlflow.log_params(self.config.all_params)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2", r2)

            if tracking_url_type_store != "file":
                mlflow.sklearn.log_model(model, "model", registered_model_name="CatboostRegressor")
            else:
                mlflow.sklearn.log_model(model, "model")

        mlflow.end_run()

