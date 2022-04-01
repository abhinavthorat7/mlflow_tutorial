from matplotlib import mlab
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from data_prepartion import data_preparation
import mlflow
import mlflow.sklearn

from fetch_data import fetch_data
import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


housing = fetch_data()

housing_prepared, housing_labels, X_test_prepared, y_test = data_preparation(housing)


if __name__ == "__main__":
    with mlflow.start_run(experiment_id=1):
        n_estimators = 90
        max_features = 10

        forest_reg = RandomForestRegressor(
            random_state=42, n_estimators=n_estimators, max_features=max_features
        )
        forest_reg.fit(housing_prepared, housing_labels)

        final_predictions = forest_reg.predict(X_test_prepared)
        mse = mean_squared_error(y_test, final_predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, final_predictions)
        print("mse", mse)
        print("rmse", rmse)
        print("mae", mae)
        mlflow.log_param(key="n_estimators", value=n_estimators)
        mlflow.log_param(key="max_features", value=max_features)
        mlflow.log_metrics({"mae": mae, "mse": mse, "rmse": rmse})
        mlflow.sklearn.log_model(forest_reg, "random_forest")
