import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import dagshub
import os
from dotenv import load_dotenv

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import joblib

# ------------------------------------
# LOAD ENV VARIABLES
# ------------------------------------
load_dotenv()

# ------------------------------------
# DAGSHUB INIT
# ------------------------------------
dagshub.init(
    repo_owner=os.getenv("DAGSHUB_USERNAME"),
    repo_name="AQI_PROJECT",
    mlflow=True
)

# ------------------------------------
# LOAD DATA
# ------------------------------------
df = pd.read_csv("data/processed_aqi_data.csv")

print("Dataset Shape:", df.shape)

# ------------------------------------
# FEATURES
# ------------------------------------
features = [
    "co",
    "no",
    "no2",
    "o3",
    "so2",
    "pm2_5",
    "pm10",
    "nh3",
    "hour",
    "day",
    "month",
    "aqi_lag_1",
    "aqi_rolling_mean",
    "aqi_change"
]

X = df[features]

# TARGET
y = df["aqi"]

# ------------------------------------
# TRAIN TEST SPLIT
# ------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ------------------------------------
# START MLFLOW EXPERIMENT
# ------------------------------------
mlflow.set_experiment("AQI_Prediction_Experiment")

with mlflow.start_run():

    # --------------------------------
    # MODEL
    # --------------------------------
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

    # Train
    model.fit(X_train, y_train)

    # Predict
    predictions = model.predict(X_test)

    # --------------------------------
    # METRICS
    # --------------------------------
    mae = mean_absolute_error(y_test, predictions)

    rmse = np.sqrt(
        mean_squared_error(y_test, predictions)
    )

    r2 = r2_score(y_test, predictions)

    # --------------------------------
    # PRINT RESULTS
    # --------------------------------
    print("\\n===== MODEL METRICS =====")

    print("MAE:", mae)
    print("RMSE:", rmse)
    print("R2:", r2)

    # --------------------------------
    # LOG PARAMETERS
    # --------------------------------
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("random_state", 42)

    # --------------------------------
    # LOG METRICS
    # --------------------------------
    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("RMSE", rmse)
    mlflow.log_metric("R2", r2)

    # --------------------------------
    # SAVE MODEL
    # --------------------------------
    joblib.dump(
        model,
        "models/mlflow_rf_model.pkl"
    )

    # --------------------------------
    # LOG MODEL
    # --------------------------------
    mlflow.sklearn.log_model(
        model,
        "RandomForestModel"
    )

    print("\\n Model logged to DagsHub MLflow")