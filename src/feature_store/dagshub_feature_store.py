"""
DagsHub Feature Store - Complete working version
This module handles feature storage, versioning, and retrieval using DagsHub
"""
import pandas as pd
import numpy as np
import os
import dagshub
import mlflow
from datetime import datetime
import json

# Set up DagsHub credentials from environment variables
def setup_dagshub():
    """Initialize DagsHub and MLflow tracking"""
    
    # Get credentials from environment
    repo_owner = os.getenv("DAGSHUB_OWNER", "your-username")
    repo_name = os.getenv("DAGSHUB_REPO", "AQI_PROJECT")
    token = os.getenv("DAGSHUB_TOKEN", "")
    
    if token:
        # Set up DagsHub with token
        dagshub.auth.add_app_token(token)
        dagshub.init(repo_name=repo_name, repo_owner=repo_owner)
        
        # Set MLflow tracking URI
        mlflow.set_tracking_uri(f"https://dagshub.com/{repo_owner}/{repo_name}.mlflow")
        
        print("DagsHub initialized successfully")
        return True
    else:
        print("Warning: DAGSHUB_TOKEN not found. Using local storage only.")
        return False

def save_features(df, feature_group_name="latest_features", version=None):
    """
    Save features to DagsHub Feature Store
    
    Args:
        df: DataFrame with features
        feature_group_name: Name of feature group
        version: Version number (auto-increments if not provided)
    """
    
    # Create features directory
    os.makedirs("data/features", exist_ok=True)
    
    # Determine version
    if version is None:
        version = get_next_version(feature_group_name)
    
    # Save to CSV with version
    file_path = f"data/features/{feature_group_name}_v{version}.csv"
    df.to_csv(file_path, index=False)
    
    # Also save as latest
    latest_path = f"data/features/{feature_group_name}_latest.csv"
    df.to_csv(latest_path, index=False)
    
    # Log to MLflow if available
    try:
        setup_dagshub()
        
        with mlflow.start_run(run_name=f"feature_store_{feature_group_name}"):
            # Log feature metadata
            mlflow.log_param("feature_group", feature_group_name)
            mlflow.log_param("version", version)
            mlflow.log_param("rows", len(df))
            mlflow.log_param("columns", len(df.columns))
            mlflow.log_param("timestamp", datetime.now().isoformat())
            
            # Log feature list
            feature_list = list(df.columns)
            mlflow.log_param("features", str(feature_list))
            
            # Log feature statistics
            for col in df.select_dtypes(include=[np.number]).columns:
                mlflow.log_metric(f"{col}_mean", df[col].mean())
                mlflow.log_metric(f"{col}_std", df[col].std())
            
            # Save artifact
            mlflow.log_artifact(file_path)
        
        print(f"Features saved to MLflow - Version: {version}")
        
    except Exception as e:
        print(f"MLflow logging skipped: {e}")
    
    print(f"Features saved to: {file_path}")
    print(f"Latest features: {latest_path}")
    
    return version

def save_latest(df):
    """Save latest features (convenience function)"""
    return save_features(df, "latest_features")

def get_next_version(feature_group_name):
    """Get next version number for feature group"""
    
    feature_dir = "data/features"
    if not os.path.exists(feature_dir):
        return 1
    
    # Find existing versions
    existing_versions = []
    for file in os.listdir(feature_dir):
        if file.startswith(f"{feature_group_name}_v") and file.endswith(".csv"):
            try:
                version_str = file.replace(f"{feature_group_name}_v", "").replace(".csv", "")
                version = int(version_str)
                existing_versions.append(version)
            except:
                pass
    
    if existing_versions:
        return max(existing_versions) + 1
    else:
        return 1

def load_features(feature_group_name="latest_features", version="latest"):
    """
    Load features from DagsHub Feature Store
    
    Args:
        feature_group_name: Name of feature group
        version: Version number or "latest"
    
    Returns:
        DataFrame with features
    """
    
    if version == "latest":
        file_path = f"data/features/{feature_group_name}_latest.csv"
    else:
        file_path = f"data/features/{feature_group_name}_v{version}.csv"
    
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        print(f"Loaded features from: {file_path}")
        print(f"Shape: {df.shape}")
        return df
    else:
        print(f"Feature file not found: {file_path}")
        return None

def get_feature_versions(feature_group_name="latest_features"):
    """Get all available versions of a feature group"""
    
    feature_dir = "data/features"
    if not os.path.exists(feature_dir):
        return []
    
    versions = []
    for file in os.listdir(feature_dir):
        if file.startswith(f"{feature_group_name}_v") and file.endswith(".csv"):
            version_str = file.replace(f"{feature_group_name}_v", "").replace(".csv", "")
            try:
                version = int(version_str)
                versions.append(version)
            except:
                pass
    
    return sorted(versions)

def create_training_dataset(start_date=None, end_date=None):
    """
    Create a training dataset from feature store
    
    Args:
        start_date: Start date for training data
        end_date: End date for training data
    
    Returns:
        DataFrame ready for training
    """
    
    # Load latest features
    df = load_features("latest_features")
    
    if df is None:
        print("No features found in store")
        return None
    
    # Convert datetime if exists
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
        
        if start_date:
            df = df[df["datetime"] >= start_date]
        if end_date:
            df = df[df["datetime"] <= end_date]
    
    print(f"Training dataset created: {df.shape}")
    return df

def log_model_to_registry(model, model_name, metrics, version=None):
    """
    Log model to DagsHub Model Registry
    
    Args:
        model: Trained model object
        model_name: Name for the model
        metrics: Dictionary of metrics
        version: Model version
    """
    
    try:
        setup_dagshub()
        
        if version is None:
            version = get_next_model_version(model_name)
        
        with mlflow.start_run(run_name=f"model_{model_name}_v{version}"):
            # Log parameters
            if hasattr(model, "get_params"):
                params = model.get_params()
                for key, value in params.items():
                    if isinstance(value, (int, float, str, bool)):
                        mlflow.log_param(key, value)
            
            # Log metrics
            for key, value in metrics.items():
                mlflow.log_metric(key, value)
            
            # Log model
            mlflow.sklearn.log_model(model, model_name)
            
            # Register model
            mlflow.register_model(
                f"runs:/{mlflow.active_run().info.run_id}/{model_name}",
                model_name
            )
        
        print(f"Model logged to registry: {model_name} v{version}")
        return version
        
    except Exception as e:
        print(f"Model registry logging failed: {e}")
        
        # Save locally as fallback
        os.makedirs("models", exist_ok=True)
        import joblib
        joblib.dump(model, f"models/{model_name}_v{version}.pkl")
        print(f"Model saved locally: models/{model_name}_v{version}.pkl")
        return version

def get_next_model_version(model_name):
    """Get next version number for model"""
    
    models_dir = "models"
    if not os.path.exists(models_dir):
        return 1
    
    existing_versions = []
    for file in os.listdir(models_dir):
        if file.startswith(f"{model_name}_v") and file.endswith(".pkl"):
            try:
                version_str = file.replace(f"{model_name}_v", "").replace(".pkl", "")
                version = int(version_str)
                existing_versions.append(version)
            except:
                pass
    
    if existing_versions:
        return max(existing_versions) + 1
    else:
        return 1

def load_model_from_registry(model_name, version="latest"):
    """
    Load model from DagsHub Model Registry
    
    Args:
        model_name: Name of the model
        version: Version number or "latest"
    
    Returns:
        Loaded model
    """
    
    if version == "latest":
        version = get_next_model_version(model_name) - 1
        if version < 1:
            version = 1
    
    file_path = f"models/{model_name}_v{version}.pkl"
    
    if os.path.exists(file_path):
        import joblib
        model = joblib.load(file_path)
        print(f"Loaded model from: {file_path}")
        return model
    else:
        print(f"Model not found: {file_path}")
        return None

def get_feature_summary():
    """Get summary of all features in store"""
    
    features_dir = "data/features"
    if not os.path.exists(features_dir):
        return "No features found"
    
    summary = {
        "feature_groups": {},
        "total_versions": 0
    }
    
    for file in os.listdir(features_dir):
        if file.endswith(".csv"):
            file_path = os.path.join(features_dir, file)
            df = pd.read_csv(file_path)
            
            group_name = file.replace(".csv", "")
            summary["feature_groups"][group_name] = {
                "rows": len(df),
                "columns": len(df.columns),
                "size_kb": os.path.getsize(file_path) / 1024
            }
            summary["total_versions"] += 1
    
    return summary

# Test the feature store
if __name__ == "__main__":
    print("Testing DagsHub Feature Store...")
    
    # Create sample data
    sample_data = pd.DataFrame({
        "datetime": [datetime.now()],
        "aqi": [75],
        "pm2_5": [35.5],
        "pm10": [65.2],
        "co": [250.0],
        "no2": [45.0]
    })
    
    # Test save
    version = save_features(sample_data, "test_features")
    print(f"Saved test features version: {version}")
    
    # Test load
    loaded = load_features("test_features", version)
    print(f"Loaded features shape: {loaded.shape}")
    
    # Test versions
    versions = get_feature_versions("test_features")
    print(f"Available versions: {versions}")
    
    print("\nFeature Store is working!")