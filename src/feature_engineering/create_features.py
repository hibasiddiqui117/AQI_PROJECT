import pandas as pd
import numpy as np

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv("data/aqi_data.csv")

# -----------------------------
# CLEAN DATA FIRST
# -----------------------------
df["datetime"] = pd.to_datetime(df["datetime"])
df = df.drop_duplicates()
df = df.dropna()

# -----------------------------
# TIME FEATURES
# -----------------------------
df["hour"] = df["datetime"].dt.hour
df["day"] = df["datetime"].dt.day
df["month"] = df["datetime"].dt.month

# -----------------------------
# SORT (IMPORTANT FOR TIME SERIES)
# -----------------------------
df = df.sort_values("datetime")

# -----------------------------
# LAG FEATURES
# -----------------------------
df["aqi_lag_1"] = df["aqi"].shift(1)
df["aqi_lag_2"] = df["aqi"].shift(2)

# -----------------------------
# ROLLING FEATURES
# -----------------------------
df["aqi_rolling_mean"] = df["aqi"].rolling(window=3).mean()
df["aqi_rolling_std"] = df["aqi"].rolling(window=3).std()

# -----------------------------
# CHANGE RATE
# -----------------------------
df["aqi_change"] = df["aqi"].diff()

# -----------------------------
# CLEAN AGAIN AFTER FEATURES
# -----------------------------
df = df.dropna()

# -----------------------------
# REALISTIC AQI CLASS CREATION
# (BASED ON PM2.5 - WORLD STANDARD)
# -----------------------------
def categorize_aqi(pm):
    if pm <= 12:
        return 1  # Good
    elif pm <= 35:
        return 2  # Moderate
    elif pm <= 55:
        return 3  # Unhealthy (Sensitive)
    elif pm <= 150:
        return 4  # Unhealthy
    else:
        return 5  # Hazardous


df["aqi_class"] = df["pm2_5"].apply(categorize_aqi)

# -----------------------------
# OPTIONAL: BALANCE DATA (SAFE VERSION)
# -----------------------------
def balance_aqi(df):
    new_rows = []

    for _, row in df.iterrows():

        for shift in [-1, 0, 1]:

            new_row = row.copy()
            new_row["aqi_class"] = max(1, min(5, row["aqi_class"] + shift))

            new_rows.append(new_row)

    return pd.DataFrame(new_rows)


df = balance_aqi(df)

# -----------------------------
# SAVE PROCESSED DATA
# -----------------------------
df.to_csv("data/processed_aqi_data.csv", index=False)

print("Feature Engineering Completed Successfully!")
print("Final dataset shape:", df.shape)
print(df["aqi_class"].value_counts())