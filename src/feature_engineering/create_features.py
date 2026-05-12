import pandas as pd

# Load data
df = pd.read_csv("data/aqi_data.csv")

# Convert datetime
df["datetime"] = pd.to_datetime(df["datetime"])

# Time Features
df["hour"] = df["datetime"].dt.hour
df["day"] = df["datetime"].dt.day
df["month"] = df["datetime"].dt.month

# Lag Features
df["aqi_lag_1"] = df["aqi"].shift(1)

# Rolling Features
df["aqi_rolling_mean"] = df["aqi"].rolling(window=3).mean()

# AQI Change Rate
df["aqi_change"] = df["aqi"].diff()

# Drop NaN values
df.dropna(inplace=True)

# Save processed features
df.to_csv("data/processed_aqi_data.csv", index=False)

print("✅ Feature Engineering Completed!")
print(df.head())