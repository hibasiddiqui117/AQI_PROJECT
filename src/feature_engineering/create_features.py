import pandas as pd

# Load data
df = pd.read_csv("data/aqi_data.csv")

# Convert datetime
df["datetime"] = pd.to_datetime(df["datetime"])

# TIME FEATURES
df["hour"] = df["datetime"].dt.hour
df["day"] = df["datetime"].dt.day
df["month"] = df["datetime"].dt.month

# LAG FEATURES (TIME SERIES MEMORY)
df["aqi_lag_1"] = df["aqi"].shift(1)


# ROLLING FEATURES (TREND)
df["aqi_rolling_mean"] = df["aqi"].rolling(window=3).mean()

# CHANGE RATE FEATURE
df["aqi_change"] = df["aqi"].diff()


# AQI CLASSIFICATION TARGET
# Based on PM2.5 (REAL WORLD LOGIC)

df["aqi_class"] = df["pm2_5"].apply(lambda x:
    1 if x <= 50 else
    2 if x <= 100 else
    3 if x <= 150 else
    4 if x <= 200 else
    5
)

# CLEAN DATA

df.dropna(inplace=True)


# SAVE PROCESSED DATASET

df.to_csv("data/processed_aqi_data.csv", index=False)

print("Feature Engineering Completed Successfully!")
print("Final dataset shape:", df.shape)
print(df.head())