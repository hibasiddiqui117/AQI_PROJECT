"""
Training Pipeline - Trains Random Forest models
"""
import pandas as pd
import numpy as np
import os
import joblib
import warnings
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

warnings.filterwarnings("ignore")

print("="*60)
print("TRAINING PIPELINE STARTING")
print("="*60)

# Load data
df = pd.read_csv("data/processed_aqi_data.csv")
print(f"Loaded data shape: {df.shape}")

# Features
features = ["co", "no", "no2", "o3", "so2", "pm2_5", "pm10", "nh3",
            "hour", "day", "month", "day_of_week", "aqi_lag_1", "aqi_change"]

available_features = [f for f in features if f in df.columns]
X = df[available_features]
y_reg = df["aqi"]
y_clf = df["aqi_class"]

print(f"Features used: {len(available_features)}")
print(f"Class distribution:\n{df['aqi_class'].value_counts().sort_index()}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_reg, test_size=0.2, random_state=42)
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(X, y_clf, test_size=0.2, random_state=42, stratify=y_clf)

# Regression - Random Forest
print("\n" + "="*40)
print("RANDOM FOREST REGRESSION")
rf = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R2 Score: {r2:.4f}")

# Regression - Ridge
print("\n" + "="*40)
print("RIDGE REGRESSION")
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
y_pred_ridge = ridge.predict(X_test)

mae_r = mean_absolute_error(y_test, y_pred_ridge)
rmse_r = np.sqrt(mean_squared_error(y_test, y_pred_ridge))
r2_r = r2_score(y_test, y_pred_ridge)

print(f"MAE: {mae_r:.4f}")
print(f"RMSE: {rmse_r:.4f}")
print(f"R2 Score: {r2_r:.4f}")

# Classification
print("\n" + "="*40)
print("CLASSIFICATION")
clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
clf.fit(X_train_clf, y_train_clf)
y_pred_clf = clf.predict(X_test_clf)

accuracy = accuracy_score(y_test_clf, y_pred_clf)
precision = precision_score(y_test_clf, y_pred_clf, average="weighted", zero_division=0)
recall = recall_score(y_test_clf, y_pred_clf, average="weighted", zero_division=0)
f1 = f1_score(y_test_clf, y_pred_clf, average="weighted", zero_division=0)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Save models
os.makedirs("models", exist_ok=True)
joblib.dump(rf, "models/regressor.pkl")
joblib.dump(clf, "models/classifier.pkl")
joblib.dump(available_features, "models/feature_columns.pkl")

# Summary
summary = {
    "timestamp": datetime.now().isoformat(),
    "data_shape": df.shape,
    "regression_r2": float(r2),
    "classification_accuracy": float(accuracy),
    "best_model": "Random Forest"
}

import json
with open("models/training_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\n" + "="*60)
print("TRAINING COMPLETED SUCCESSFULLY")
print("="*60)