"""
Training Pipeline - Completely fixed version
"""
import pandas as pd
import numpy as np
import os
import joblib
import warnings
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Suppress warnings
warnings.filterwarnings("ignore")

print("="*60)
print("TRAINING PIPELINE STARTING")
print("="*60)

def load_and_prepare_data(file_path):
    """Load and prepare data for training"""
    
    if not os.path.exists(file_path):
        print(f"Error: File not found - {file_path}")
        return None
    
    df = pd.read_csv(file_path)
    print(f"Loaded data shape: {df.shape}")
    
    # Ensure datetime is proper
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    
    # Create time features if missing
    if "hour" not in df.columns and "datetime" in df.columns:
        df["hour"] = df["datetime"].dt.hour
        df["day"] = df["datetime"].dt.day
        df["month"] = df["datetime"].dt.month
        df["day_of_week"] = df["datetime"].dt.dayofweek
        print("Added time features")
    
    # Create lag features if missing
    if "aqi_lag_1" not in df.columns:
        df = df.sort_values("datetime").reset_index(drop=True)
        df["aqi_lag_1"] = df["aqi"].shift(1)
        df["aqi_lag_3"] = df["aqi"].shift(3)
        df["aqi_rolling_mean"] = df["aqi"].rolling(window=6, min_periods=1).mean()
        df["aqi_change"] = df["aqi"].diff().fillna(0)
        print("Added lag features")
    
    # Drop rows with NaN
    before = len(df)
    df = df.dropna().reset_index(drop=True)
    after = len(df)
    if before != after:
        print(f"Removed {before - after} rows with NaN")
    
    return df

# Feature columns to use
FEATURE_COLUMNS = [
    "co", "no", "no2", "o3", "so2", "pm2_5", "pm10", "nh3",
    "hour", "day", "month", "day_of_week",
    "aqi_lag_1", "aqi_rolling_mean", "aqi_change"
]

# Load data
df = load_and_prepare_data("data/processed_aqi_data.csv")

if df is None or len(df) < 50:
    print("Trying alternative path...")
    df = load_and_prepare_data("data/aqi_data.csv")

if df is None or len(df) < 50:
    print("Fatal Error: No valid data found!")
    exit(1)

# Check which features are available
available_features = [f for f in FEATURE_COLUMNS if f in df.columns]
missing_features = [f for f in FEATURE_COLUMNS if f not in df.columns]

if missing_features:
    print(f"Warning: Missing features - {missing_features}")

X = df[available_features]

# Create AQI class if not exists
if "aqi_class" not in df.columns:
    def get_class(aqi):
        if aqi <= 50: return 1
        elif aqi <= 100: return 2
        elif aqi <= 150: return 3
        elif aqi <= 200: return 4
        elif aqi <= 300: return 5
        else: return 6
    
    df["aqi_class"] = df["aqi"].apply(get_class)
    print("Added AQI class column")

# Define class names
class_names = {
    1: "Good",
    2: "Moderate", 
    3: "Unhealthy Sensitive",
    4: "Unhealthy",
    5: "Very Unhealthy",
    6: "Hazardous"
}

# Check class distribution
print("\n" + "="*60)
print("DATA DISTRIBUTION")
print("="*60)
print(f"Total samples: {len(df)}")
print(f"Features: {len(available_features)}")
print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")

print("\nAQI Class Distribution:")
class_dist = df["aqi_class"].value_counts().sort_index()

for cls in sorted(class_dist.keys()):
    count = class_dist[cls]
    name = class_names.get(int(cls), f"Class {cls}")
    bar_length = min(count // 2, 30)
    bar = "*" * bar_length
    print(f"  {name:22} : {count:3} samples {bar}")

# Create models directory
os.makedirs("models", exist_ok=True)

# ============================================
# REGRESSION MODEL (Predict AQI)
# ============================================
print("\n" + "="*60)
print("REGRESSION MODEL TRAINING")
print("="*60)

y_reg = df["aqi"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y_reg, test_size=0.2, random_state=42, shuffle=True
)

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# Train Random Forest Regressor
reg_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

reg_model.fit(X_train, y_train)

# Predict and evaluate
y_pred_reg = reg_model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred_reg)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_reg))
r2 = r2_score(y_test, y_pred_reg)

print("\nREGRESSION RESULTS:")
print(f"  Mean Absolute Error (MAE): {mae:.4f}")
print(f"  Root Mean Square Error (RMSE): {rmse:.4f}")
print(f"  R-squared (R2) Score: {r2:.4f}")

# Feature importance
feature_importance = pd.DataFrame({
    "feature": available_features,
    "importance": reg_model.feature_importances_
}).sort_values("importance", ascending=False)

print("\nTop 10 Most Important Features (Regression):")
for idx, row in feature_importance.head(10).iterrows():
    print(f"  {row['feature']:20}: {row['importance']:.4f}")

# Save regression model
joblib.dump(reg_model, "models/regressor.pkl")
print("\nRegression model saved: models/regressor.pkl")

# ============================================
# CLASSIFICATION MODEL (Predict AQI Class)
# ============================================
print("\n" + "="*60)
print("CLASSIFICATION MODEL TRAINING")
print("="*60)

y_clf = df["aqi_class"]

# Convert to int if needed
if y_clf.dtype == 'float64':
    y_clf = y_clf.astype(int)

# Get unique classes in data
unique_classes = sorted(y_clf.unique())
print(f"Unique classes in data: {unique_classes}")

if len(unique_classes) < 2:
    print("Warning: Only one class found. Classification model will not be trained.")
    # Create dummy classifier
    from sklearn.dummy import DummyClassifier
    clf_model = DummyClassifier(strategy="constant", constant=unique_classes[0])
    clf_model.fit(X, y_clf)
    joblib.dump(clf_model, "models/classifier.pkl")
    print("Dummy classifier saved: models/classifier.pkl")
    
else:
    # Use stratified split
    X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
        X, y_clf, test_size=0.2, random_state=42, shuffle=True, stratify=y_clf
    )
    
    print(f"Training samples: {len(X_train_clf)}")
    print(f"Test samples: {len(X_test_clf)}")
    
    # Train Random Forest Classifier
    clf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    
    clf_model.fit(X_train_clf, y_train_clf)
    
    # Predict and evaluate
    y_pred_clf = clf_model.predict(X_test_clf)
    
    accuracy = accuracy_score(y_test_clf, y_pred_clf)
    precision = precision_score(y_test_clf, y_pred_clf, average="weighted", zero_division=0)
    recall = recall_score(y_test_clf, y_pred_clf, average="weighted", zero_division=0)
    f1 = f1_score(y_test_clf, y_pred_clf, average="weighted", zero_division=0)
    
    print("\nCLASSIFICATION RESULTS:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision (weighted): {precision:.4f}")
    print(f"  Recall (weighted): {recall:.4f}")
    print(f"  F1 Score (weighted): {f1:.4f}")
    
    # Get classes present in test data (fix for numpy array)
    test_unique = np.unique(y_test_clf)
    pred_unique = np.unique(y_pred_clf)
    test_classes = sorted(list(set(test_unique) | set(pred_unique)))
    
    # Convert to int if needed
    test_classes = [int(c) for c in test_classes]
    
    # Get class names for display
    test_class_names = [class_names.get(c, f"Class {c}") for c in test_classes]
    
    print("\nClassification Report:")
    print(classification_report(
        y_test_clf, y_pred_clf, 
        labels=test_classes,
        target_names=test_class_names, 
        zero_division=0
    ))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test_clf, y_pred_clf, labels=test_classes)
    cm_df = pd.DataFrame(cm, index=test_class_names, columns=test_class_names)
    print(cm_df)
    
    # Feature importance for classification
    clf_feature_importance = pd.DataFrame({
        "feature": available_features,
        "importance": clf_model.feature_importances_
    }).sort_values("importance", ascending=False)
    
    print("\nTop 10 Most Important Features (Classification):")
    for idx, row in clf_feature_importance.head(10).iterrows():
        print(f"  {row['feature']:20}: {row['importance']:.4f}")
    
    # Save classification model
    joblib.dump(clf_model, "models/classifier.pkl")
    print("\nClassification model saved: models/classifier.pkl")

# Save feature columns
joblib.dump(available_features, "models/feature_columns.pkl")
print("Feature columns saved: models/feature_columns.pkl")

# Save training summary
summary = {
    "timestamp": datetime.now().isoformat(),
    "data_shape": df.shape,
    "features": available_features,
    "features_count": len(available_features),
    "regression_mae": float(mae),
    "regression_rmse": float(rmse),
    "regression_r2": float(r2),
    "unique_classes": [int(c) for c in unique_classes],
    "class_distribution": {int(k): int(v) for k, v in class_dist.items()}
}

if len(unique_classes) >= 2:
    summary["classification_accuracy"] = float(accuracy)
    summary["classification_f1"] = float(f1)

import json
with open("models/training_summary.json", "w") as f:
    json.dump(summary, f, indent=2, default=str)

print("\nTraining summary saved: models/training_summary.json")

print("\n" + "="*60)
print("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
print("="*60)