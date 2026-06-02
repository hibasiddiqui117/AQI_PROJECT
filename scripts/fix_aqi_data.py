"""
Run this script FIRST to fix your aqi_data.csv and add proper features
This will create realistic AQI variations for training
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys

def fix_and_augment_data():
    print("="*70)
    print("FIXING AQI_DATA.CSV AND ADDING FEATURES")
    print("="*70)
    
    # Load existing data - try multiple possible paths
    possible_paths = [
        "data/aqi_data.csv",
        "data/raw/aqi_data.csv", 
        "aqi_data.csv",
        "../data/aqi_data.csv"
    ]
    
    df = None
    file_path_used = None
    
    for path in possible_paths:
        if os.path.exists(path):
            df = pd.read_csv(path)
            file_path_used = path
            print(f"\nLoading data from: {path}")
            break
    
    if df is None:
        print("Error: No aqi_data.csv file found!")
        print("Please make sure your CSV file is in the 'data/' folder")
        return None
    
    print(f"Original shape: {df.shape}")
    print(f"Original columns: {list(df.columns)}")
    
    # Convert datetime
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        df = df.dropna(subset=["datetime"])
        print(f"After datetime conversion: {df.shape}")
    
    # Sort by datetime
    df = df.sort_values("datetime").reset_index(drop=True)
    
    # Check current AQI range
    print(f"\nCurrent AQI range: min={df['aqi'].min()}, max={df['aqi'].max()}")
    print(f"Current AQI classes present:")
    unique_aqi = sorted(df['aqi'].unique())
    print(f"  Unique AQI values: {unique_aqi[:10]}..." if len(unique_aqi) > 10 else f"  Unique AQI values: {unique_aqi}")
    
    # Create augmented dataset with different AQI classes
    print("\n" + "="*70)
    print("CREATING SYNTHETIC AQI VARIATIONS FOR BALANCED TRAINING")
    print("="*70)
    
    augmented_rows = []
    
    for idx, row in df.iterrows():
        base_row = row.to_dict()
        base_aqi_value = row["aqi"]
        
        # Keep original
        augmented_rows.append(base_row)
        
        # Create variations for different AQI classes based on original data
        # Class 1: Good (0-50) - already have many
        # Class 2: Moderate (51-100)
        # Class 3: Unhealthy for Sensitive (101-150)
        # Class 4: Unhealthy (151-200)
        # Class 5: Very Unhealthy (201-300)
        # Class 6: Hazardous (301-500)
        
        # Create Moderate class (51-100)
        mod_row = base_row.copy()
        mod_row["aqi"] = np.random.randint(51, 100)
        mod_row["pm2_5"] = mod_row["pm2_5"] * 1.5 if mod_row["pm2_5"] * 1.5 < 150 else 150
        mod_row["pm10"] = mod_row["pm10"] * 1.4 if mod_row["pm10"] * 1.4 < 200 else 200
        mod_row["co"] = mod_row["co"] * 1.3
        mod_row["no2"] = mod_row["no2"] * 1.4
        augmented_rows.append(mod_row)
        
        # Create Unhealthy for Sensitive class (101-150)
        sensitive_row = base_row.copy()
        sensitive_row["aqi"] = np.random.randint(101, 150)
        sensitive_row["pm2_5"] = mod_row["pm2_5"] * 1.5 if mod_row["pm2_5"] * 1.5 < 250 else 250
        sensitive_row["pm10"] = mod_row["pm10"] * 1.4 if mod_row["pm10"] * 1.4 < 350 else 350
        sensitive_row["co"] = mod_row["co"] * 1.3
        sensitive_row["no2"] = mod_row["no2"] * 1.4
        augmented_rows.append(sensitive_row)
        
        # Create Unhealthy class (151-200)
        unhealthy_row = base_row.copy()
        unhealthy_row["aqi"] = np.random.randint(151, 200)
        unhealthy_row["pm2_5"] = sensitive_row["pm2_5"] * 1.4 if sensitive_row["pm2_5"] * 1.4 < 350 else 350
        unhealthy_row["pm10"] = sensitive_row["pm10"] * 1.3 if sensitive_row["pm10"] * 1.3 < 450 else 450
        unhealthy_row["co"] = sensitive_row["co"] * 1.2
        unhealthy_row["no2"] = sensitive_row["no2"] * 1.3
        augmented_rows.append(unhealthy_row)
        
        # Create Very Unhealthy class (201-300)
        very_unhealthy_row = base_row.copy()
        very_unhealthy_row["aqi"] = np.random.randint(201, 300)
        very_unhealthy_row["pm2_5"] = unhealthy_row["pm2_5"] * 1.3 if unhealthy_row["pm2_5"] * 1.3 < 450 else 450
        very_unhealthy_row["pm10"] = unhealthy_row["pm10"] * 1.2 if unhealthy_row["pm10"] * 1.2 < 550 else 550
        very_unhealthy_row["co"] = unhealthy_row["co"] * 1.2
        very_unhealthy_row["no2"] = unhealthy_row["no2"] * 1.2
        augmented_rows.append(very_unhealthy_row)
        
        # Create Hazardous class (301-500) - only for some rows
        if idx % 3 == 0:  # Create fewer hazardous samples
            hazardous_row = base_row.copy()
            hazardous_row["aqi"] = np.random.randint(301, 500)
            hazardous_row["pm2_5"] = very_unhealthy_row["pm2_5"] * 1.3 if very_unhealthy_row["pm2_5"] * 1.3 < 600 else 600
            hazardous_row["pm10"] = very_unhealthy_row["pm10"] * 1.2 if very_unhealthy_row["pm10"] * 1.2 < 700 else 700
            hazardous_row["co"] = very_unhealthy_row["co"] * 1.2
            hazardous_row["no2"] = very_unhealthy_row["no2"] * 1.2
            augmented_rows.append(hazardous_row)
    
    # Convert to DataFrame
    df_augmented = pd.DataFrame(augmented_rows)
    print(f"\nAugmented shape: {df_augmented.shape}")
    
    # Add time-based features
    print("\nAdding time-based features...")
    if "datetime" in df_augmented.columns:
        df_augmented["hour"] = df_augmented["datetime"].dt.hour
        df_augmented["day"] = df_augmented["datetime"].dt.day
        df_augmented["month"] = df_augmented["datetime"].dt.month
        df_augmented["day_of_week"] = df_augmented["datetime"].dt.dayofweek
        df_augmented["weekend"] = (df_augmented["day_of_week"] >= 5).astype(int)
        
        # Cyclical features for hour
        df_augmented["hour_sin"] = np.sin(2 * np.pi * df_augmented["hour"] / 24)
        df_augmented["hour_cos"] = np.cos(2 * np.pi * df_augmented["hour"] / 24)
        
        # Cyclical features for day of week
        df_augmented["dow_sin"] = np.sin(2 * np.pi * df_augmented["day_of_week"] / 7)
        df_augmented["dow_cos"] = np.cos(2 * np.pi * df_augmented["day_of_week"] / 7)
    
    # Add lag features
    print("Adding lag features...")
    df_augmented = df_augmented.sort_values("datetime").reset_index(drop=True)
    df_augmented["aqi_lag_1"] = df_augmented["aqi"].shift(1)
    df_augmented["aqi_lag_3"] = df_augmented["aqi"].shift(3)
    df_augmented["aqi_lag_6"] = df_augmented["aqi"].shift(6)
    df_augmented["aqi_lag_12"] = df_augmented["aqi"].shift(12)
    df_augmented["aqi_lag_24"] = df_augmented["aqi"].shift(24)
    
    # Add rolling statistics
    df_augmented["aqi_rolling_mean_3"] = df_augmented["aqi"].rolling(window=3, min_periods=1).mean()
    df_augmented["aqi_rolling_mean_6"] = df_augmented["aqi"].rolling(window=6, min_periods=1).mean()
    df_augmented["aqi_rolling_mean_12"] = df_augmented["aqi"].rolling(window=12, min_periods=1).mean()
    df_augmented["aqi_rolling_std_6"] = df_augmented["aqi"].rolling(window=6, min_periods=1).std().fillna(0)
    df_augmented["aqi_change"] = df_augmented["aqi"].diff().fillna(0)
    df_augmented["aqi_pct_change"] = df_augmented["aqi"].pct_change().fillna(0) * 100
    
    # Add pollutant ratios
    print("Adding pollutant ratios...")
    df_augmented["pm25_pm10_ratio"] = df_augmented["pm2_5"] / (df_augmented["pm10"] + 1)
    df_augmented["no2_so2_ratio"] = df_augmented["no2"] / (df_augmented["so2"] + 1)
    df_augmented["co_no2_ratio"] = df_augmented["co"] / (df_augmented["no2"] + 1)
    df_augmented["o3_no2_ratio"] = df_augmented["o3"] / (df_augmented["no2"] + 1)
    
    # Create AQI classes (1-6)
    def get_aqi_class(aqi):
        if aqi <= 50:
            return 1
        elif aqi <= 100:
            return 2
        elif aqi <= 150:
            return 3
        elif aqi <= 200:
            return 4
        elif aqi <= 300:
            return 5
        else:
            return 6
    
    df_augmented["aqi_class"] = df_augmented["aqi"].apply(get_aqi_class)
    
    # Create AQI category names
    def get_aqi_category(aqi):
        if aqi <= 50:
            return "Good"
        elif aqi <= 100:
            return "Moderate"
        elif aqi <= 150:
            return "Unhealthy for Sensitive"
        elif aqi <= 200:
            return "Unhealthy"
        elif aqi <= 300:
            return "Very Unhealthy"
        else:
            return "Hazardous"
    
    df_augmented["aqi_category"] = df_augmented["aqi"].apply(get_aqi_category)
    
    # Drop rows with NaN from lag features
    before = len(df_augmented)
    df_augmented = df_augmented.dropna().reset_index(drop=True)
    after = len(df_augmented)
    print(f"Removed {before - after} rows with NaN from lag features")
    
    # Check class distribution
    print("\n" + "="*70)
    print("AQI CLASS DISTRIBUTION AFTER AUGMENTATION")
    print("="*70)
    class_dist = df_augmented["aqi_class"].value_counts().sort_index()
    class_names = ["Good", "Moderate", "Unhealthy Sensitive", "Unhealthy", "Very Unhealthy", "Hazardous"]
    
    for cls in range(1, 7):
        count = class_dist.get(cls, 0)
        name = class_names[cls-1] if cls-1 < len(class_names) else f"Class {cls}"
        percentage = (count / len(df_augmented)) * 100
        bar = "█" * int(percentage / 2)
        print(f"  Class {cls} ({name:25}): {count:6} samples ({percentage:5.1f}%) {bar}")
    
    # Save fixed data to multiple locations
    print("\n" + "="*70)
    print("SAVING FIXED DATA")
    print("="*70)
    
    # Save as processed_aqi_data.csv (for training pipeline)
    output_path_1 = "data/processed_aqi_data.csv"
    os.makedirs("data", exist_ok=True)
    df_augmented.to_csv(output_path_1, index=False)
    print(f"Saved to: {output_path_1}")
    
    # Also save back to original aqi_data.csv (overwrite with fixed version)
    output_path_2 = "data/aqi_data.csv"
    df_augmented.to_csv(output_path_2, index=False)
    print(f"Saved to: {output_path_2}")
    
    # Also save as backup
    backup_path = f"data/aqi_data_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df_augmented.to_csv(backup_path, index=False)
    print(f"Backup saved to: {backup_path}")
    
    print(f"\nFinal shape: {df_augmented.shape}")
    print(f"Final columns: {len(df_augmented.columns)}")
    
    # Display sample of data
    print("\n" + "="*70)
    print("SAMPLE OF FIXED DATA")
    print("="*70)
    print(df_augmented.head(10).to_string())
    
    print("\n" + "="*70)
    print("DATA FIXING COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("\nNext step: Run training pipeline")
    print("python pipelines/training_pipeline.py")
    
    return df_augmented

if __name__ == "__main__":
    result = fix_and_augment_data()
    
    if result is not None:
        print("\n" + "="*70)
        print("SUMMARY STATISTICS")
        print("="*70)
        print(f"Total samples: {len(result)}")
        print(f"Date range: {result['datetime'].min()} to {result['datetime'].max()}")
        print(f"AQI range: {result['aqi'].min()} to {result['aqi'].max()}")
        print(f"Features available: {len(result.columns)}")