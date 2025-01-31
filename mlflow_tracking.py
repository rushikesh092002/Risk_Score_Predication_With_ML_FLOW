import mlflow
import dagshub

dagshub.init(
  repo_owner='rushikesh092002', 
  repo_name='Risk_Score_Predication_With_ML_FLOW', 
  mlflow=True)

mlflow.set_tracking_uri("https://dagshub.com/rushikesh092002/Risk_Score_Predication_With_ML_FLOW.mlflow")

with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)