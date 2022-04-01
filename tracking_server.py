import mlflow
import mlflow.sklearn

# remote_server_uri = " /home/abhinav_thorat/mle-training/"
mlflow.set_tracking_uri("file:///home/abhinav_thorat/mle-training/mlruns")
print(mlflow.tracking.get_tracking_uri())


exp_name = "House_price"
mlflow.set_experiment(exp_name)
