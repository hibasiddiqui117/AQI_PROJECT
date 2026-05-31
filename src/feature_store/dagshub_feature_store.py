import pandas as pd
import os

STORE_PATH = "data/feature_store"

os.makedirs(STORE_PATH, exist_ok=True)

# -----------------------------
# SAVE FEATURES (VERSIONED)
# -----------------------------
def save_features(df, version="v1"):

    path = f"{STORE_PATH}/features_{version}.csv"
    df.to_csv(path, index=False)

    print(f"Saved Feature Store: {path}")

# -----------------------------
# LOAD FEATURES
# -----------------------------
def load_features(version="v1"):

    path = f"{STORE_PATH}/features_{version}.csv"
    return pd.read_csv(path)

# -----------------------------
# SAVE LATEST (AUTO UPDATE)
# -----------------------------
def save_latest(df):

    path = f"{STORE_PATH}/latest_features.csv"
    df.to_csv(path, index=False)

    print("Updated Latest Feature Store")