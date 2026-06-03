"""
Feature Pipeline for GitHub Actions - No DagsHub dependency
This version works without dagshub module
"""
import requests
import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys
import traceback
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

print("="*60)
print("FEATURE PIPELINE (GitHub Actions Version)")
print("="*60)

# Configuration
API_KEY = os.getenv("OPENWEATHER_API_KEY")
LAT = float(os.getenv("LAT", 24.8607))
LON = float(os.getenv("LON", 67.0011))

print(f"API Key present: {'Yes' if API_KEY else 'No'}")
print(f"Coordinates: Lat={LAT}, Lon={LON}")

if not API_KEY:
    print("\nERROR: OPENWEATHER_API_KEY missing!")
    sys.exit(1)

# API call
url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={LAT}&lon={LON}&appid={API_KEY}"

print(f"\nFetching data from API...")

try:
    response = requests.get(url, timeout=30)
    print(f"Response status: {response.status_code}")
    
    if response.status_code != 200:
        print(f"API Error: {response.text}")
        sys.exit(1)
    
    data = response.json()
    pollution = data["list"][0]
    
    aqi = pollution["main"]["aqi"]
    components = pollution["components"]
    
    print(f"\nAPI Success - Current AQI: {aqi}")
    print(f"PM2.5: {components['pm2_5']:.1f}, PM10: {components['pm10']:.1f}")
    
    # Create new row
    current_time = datetime.now()
    
    row = {
        "datetime": current_time,
        "aqi": aqi,
        "co": components["co"],
        "no": components["no"],
        "no2": components["no2"],
        "o3": components["o3"],
        "so2": components["so2"],
        "pm2_5": components["pm2_5"],
        "pm10": components["pm10"],
        "nh3": components["nh3"]
    }
    
    df_new = pd.DataFrame([row])
    
    # Create data directory
    os.makedirs("data", exist_ok=True)
    file_path = "data/aqi_data.csv"
    
    # Load existing data
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        df_existing = pd.read_csv(file_path)
        df_existing["datetime"] = pd.to_datetime(df_existing["datetime"])
        df_new["datetime"] = pd.to_datetime(df_new["datetime"])
        
        # Combine and remove duplicates
        df_final = pd.concat([df_existing, df_new], ignore_index=True)
        df_final = df_final.drop_duplicates(subset=["datetime"], keep="last")
        df_final = df_final.sort_values("datetime").reset_index(drop=True)
        
        print(f"Combined: {len(df_existing)} + 1 = {len(df_final)} records")
    else:
        df_final = df_new
        print("Created new dataset")
    
    # Save raw data
    df_final.to_csv(file_path, index=False)
    print(f"Raw data saved: {file_path}")
    
    # ============================================
    # FEATURE ENGINEERING
    # ============================================
    print("\n" + "="*40)
    print("FEATURE ENGINEERING")
    print("="*40)
    
    df_processed = df_final.copy()
    
    # Time-based features
    df_processed["hour"] = df_processed["datetime"].dt.hour
    df_processed["day"] = df_processed["datetime"].dt.day
    df_processed["month"] = df_processed["datetime"].dt.month
    df_processed["day_of_week"] = df_processed["datetime"].dt.dayofweek
    
    # Cyclical encoding for hour
    df_processed["hour_sin"] = np.sin(2 * np.pi * df_processed["hour"] / 24)
    df_processed["hour_cos"] = np.cos(2 * np.pi * df_processed["hour"] / 24)
    
    # Lag features (AQI change rate features)
    df_processed = df_processed.sort_values("datetime").reset_index(drop=True)
    df_processed["aqi_lag_1"] = df_processed["aqi"].shift(1)
    df_processed["aqi_lag_3"] = df_processed["aqi"].shift(3)
    df_processed["aqi_lag_6"] = df_processed["aqi"].shift(6)
    df_processed["aqi_lag_12"] = df_processed["aqi"].shift(12)
    df_processed["aqi_lag_24"] = df_processed["aqi"].shift(24)
    
    # AQI change rate (derived feature)
    df_processed["aqi_change"] = df_processed["aqi"].diff().fillna(0)
    df_processed["aqi_pct_change"] = (df_processed["aqi"].pct_change() * 100).fillna(0)
    
    # Rolling statistics
    df_processed["aqi_rolling_mean_3"] = df_processed["aqi"].rolling(window=3, min_periods=1).mean()
    df_processed["aqi_rolling_mean_6"] = df_processed["aqi"].rolling(window=6, min_periods=1).mean()
    df_processed["aqi_rolling_mean_12"] = df_processed["aqi"].rolling(window=12, min_periods=1).mean()
    df_processed["aqi_rolling_std_6"] = df_processed["aqi"].rolling(window=6, min_periods=1).std().fillna(0)
    
    # Pollutant ratios (derived features)
    df_processed["pm25_pm10_ratio"] = df_processed["pm2_5"] / (df_processed["pm10"] + 1)
    df_processed["no2_so2_ratio"] = df_processed["no2"] / (df_processed["so2"] + 1)
    df_processed["co_no2_ratio"] = df_processed["co"] / (df_processed["no2"] + 1)
    
    # AQI class for classification
    def get_aqi_class(aqi_val):
        if aqi_val <= 50: return 1
        elif aqi_val <= 100: return 2
        elif aqi_val <= 150: return 3
        elif aqi_val <= 200: return 4
        elif aqi_val <= 300: return 5
        else: return 6
    
    df_processed["aqi_class"] = df_processed["aqi"].apply(get_aqi_class)
    
    # Drop rows with NaN from lag features
    before = len(df_processed)
    df_processed = df_processed.dropna().reset_index(drop=True)
    after = len(df_processed)
    
    if before != after:
        print(f"Removed {before - after} rows with NaN values")
    
    # Save processed data
    processed_path = "data/processed_aqi_data.csv"
    df_processed.to_csv(processed_path, index=False)
    print(f"Processed data saved: {processed_path}")
    print(f"Total records: {len(df_processed)}")
    print(f"Features created: {len(df_processed.columns)}")
    
    print("\n" + "="*60)
    print("FEATURE PIPELINE COMPLETED SUCCESSFULLY")
    print("="*60)
    
except Exception as e:
    print(f"\nError: {e}")
    traceback.print_exc()
    sys.exit(1)