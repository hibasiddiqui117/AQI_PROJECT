import pandas as pd
import os

FEATURE_STORE_PATH = "data/feature_store"

os.makedirs(FEATURE_STORE_PATH, exist_ok=True)

def save_features(df, version_name):
    
    path = f"{FEATURE_STORE_PATH}/{version_name}.csv"

    df.to_csv(path, index=False)

    print(f"Saved: {path}")

def load_features(version_name):

    path = f"{FEATURE_STORE_PATH}/{version_name}.csv"

    return pd.read_csv(path)