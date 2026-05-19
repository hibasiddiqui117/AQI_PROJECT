import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split

# Models
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

# Metrics (STRICTLY CORRECT)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# LOAD DATA
df = pd.read_csv("data/processed_aqi_data.csv")

print("Dataset Shape:", df.shape)

# FEATURES
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

os.makedirs("models", exist_ok=True)


# REGRESSION MODEL (AQI / PM2.5 prediction)
y_reg = df["pm2_5"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y_reg, test_size=0.2, random_state=42
)

reg_model = RandomForestRegressor(n_estimators=100, random_state=42)
reg_model.fit(X_train, y_train)

pred_reg = reg_model.predict(X_test)

print("\n===== REGRESSION RESULTS =====")
print("MAE:", mean_absolute_error(y_test, pred_reg))
print("RMSE:", mean_squared_error(y_test, pred_reg) ** 0.5)
print("R2 Score:", r2_score(y_test, pred_reg))

joblib.dump(reg_model, "models/regressor.pkl")

# CLASSIFICATION MODEL (AQI CLASS)
y_clf = df["aqi_class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y_clf, test_size=0.2, random_state=42
)

clf_model = RandomForestClassifier(n_estimators=100, random_state=42)
clf_model.fit(X_train, y_train)

pred_clf = clf_model.predict(X_test)

print("\n===== CLASSIFICATION RESULTS =====")
print("Accuracy:", accuracy_score(y_test, pred_clf))
print("Precision:", precision_score(y_test, pred_clf, average="weighted"))
print("Recall:", recall_score(y_test, pred_clf, average="weighted"))
print("F1 Score:", f1_score(y_test, pred_clf, average="weighted"))

print("\nConfusion Matrix:\n", confusion_matrix(y_test, pred_clf))

joblib.dump(clf_model, "models/classifier.pkl")

print("\n Training Pipeline Completed Successfully!")