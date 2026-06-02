"""
Run this script once to clean your existing CSV data
This will remove duplicates and fix inconsistencies
"""
import pandas as pd
import numpy as np
import os
import sys

def clean_existing_csv():
    print("="*60)
    print("DATA CLEANING SCRIPT")
    print("="*60)
    
    # Create necessary directories
    os.makedirs("data", exist_ok=True)
    os.makedirs("scripts", exist_ok=True)
    
    # Load existing data - try multiple possible locations
    file_paths = [
        "data/aqi_data.csv",
        "data/raw/aqi_data.csv", 
        "data/processed_aqi_data.csv",
        "aqi_data.csv"
    ]
    
    df = None
    loaded_path = None
    
    for path in file_paths:
        if os.path.exists(path):
            print(f"\nFound data file: {path}")
            df = pd.read_csv(path)
            loaded_path = path
            break
    
    if df is None:
        print("\nError: No data file found in any location!")
        print("Searched in:", file_paths)
        print("\nPlease make sure aqi_data.csv exists in the data/ folder")
        return False
    
    print(f"\nLoading data from: {loaded_path}")
    print(f"Original shape: {df.shape}")
    print(f"Original columns: {list(df.columns)}")
    
    # Display first few rows for debugging
    print("\nFirst 5 rows of raw data:")
    print(df.head())
    
    # ============================================
    # STEP 1: Handle datetime column
    # ============================================
    print("\n" + "-"*40)
    print("STEP 1: Cleaning datetime column")
    print("-"*40)
    
    # Find datetime column (could be named differently)
    datetime_col = None
    for col in df.columns:
        if 'datetime' in col.lower() or 'date' in col.lower() or 'time' in col.lower():
            datetime_col = col
            break
    
    if datetime_col is None:
        print("Warning: No datetime column found. Creating from index.")
        df["datetime"] = pd.date_range(start="2024-01-01", periods=len(df), freq="H")
        datetime_col = "datetime"
    else:
        print(f"Using datetime column: {datetime_col}")
        df[datetime_col] = pd.to_datetime(df[datetime_col], errors="coerce")
    
    # Remove rows with invalid datetime
    before = len(df)
    df = df.dropna(subset=[datetime_col])
    after = len(df)
    if before != after:
        print(f"Removed {before - after} rows with invalid datetime")
    
    # Rename to standard datetime if needed
    if datetime_col != "datetime":
        df = df.rename(columns={datetime_col: "datetime"})
    
    # ============================================
    # STEP 2: Remove duplicates
    # ============================================
    print("\n" + "-"*40)
    print("STEP 2: Removing duplicate records")
    print("-"*40)
    
    before = len(df)
    df = df.drop_duplicates(subset=["datetime"], keep="last")
    after = len(df)
    if before != after:
        print(f"Removed {before - after} duplicate rows based on datetime")
    
    # Sort by datetime
    df = df.sort_values("datetime").reset_index(drop=True)
    
    # ============================================
    # STEP 3: Remove outliers
    # ============================================
    print("\n" + "-"*40)
    print("STEP 3: Removing outliers")
    print("-"*40)
    
    # List of columns to check for outliers
    outlier_columns = ["aqi", "pm2_5", "pm10", "co", "no", "no2", "o3", "so2", "nh3"]
    
    before = len(df)
    
    for col in outlier_columns:
        if col in df.columns:
            # Define reasonable ranges for each pollutant
            if col == "aqi":
                df = df[(df[col] >= 0) & (df[col] <= 500)]
            elif col in ["pm2_5", "pm10"]:
                df = df[(df[col] >= 0) & (df[col] <= 500)]
            elif col == "co":
                df = df[(df[col] >= 0) & (df[col] <= 1000)]
            elif col in ["no", "no2"]:
                df = df[(df[col] >= 0) & (df[col] <= 200)]
            elif col == "o3":
                df = df[(df[col] >= 0) & (df[col] <= 200)]
            elif col == "so2":
                df = df[(df[col] >= 0) & (df[col] <= 100)]
            elif col == "nh3":
                df = df[(df[col] >= 0) & (df[col] <= 50)]
    
    after = len(df)
    if before != after:
        print(f"Removed {before - after} rows with outlier values")
    
    # ============================================
    # STEP 4: Handle missing values
    # ============================================
    print("\n" + "-"*40)
    print("STEP 4: Handling missing values")
    print("-"*40)
    
    # Check missing values
    missing_counts = df.isnull().sum()
    missing_cols = missing_counts[missing_counts > 0]
    
    if len(missing_cols) > 0:
        print("Missing values found in columns:")
        for col, count in missing_cols.items():
            print(f"  {col}: {count} missing values")
        
        # Fill missing values with median or forward fill
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                if col in ["aqi", "pm2_5", "pm10"]:
                    # Use forward fill for important columns
                    df[col] = df[col].fillna(method="ffill").fillna(method="bfill")
                else:
                    # Use median for other columns
                    df[col] = df[col].fillna(df[col].median())
        
        print("Missing values filled")
    
    # ============================================
    # STEP 5: Add time-based features
    # ============================================
    print("\n" + "-"*40)
    print("STEP 5: Adding time-based features")
    print("-"*40)
    
    if "hour" not in df.columns:
        df["hour"] = df["datetime"].dt.hour
        print("Added hour feature")
    
    if "day" not in df.columns:
        df["day"] = df["datetime"].dt.day
        print("Added day feature")
    
    if "month" not in df.columns:
        df["month"] = df["datetime"].dt.month
        print("Added month feature")
    
    if "day_of_week" not in df.columns:
        df["day_of_week"] = df["datetime"].dt.dayofweek
        print("Added day_of_week feature")
    
    # Add cyclical time features
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    print("Added cyclical hour features")
    
    # ============================================
    # STEP 6: Add lag features
    # ============================================
    print("\n" + "-"*40)
    print("STEP 6: Adding lag features for time series")
    print("-"*40)
    
    # Sort by datetime to ensure proper lag calculation
    df = df.sort_values("datetime").reset_index(drop=True)
    
    if "aqi_lag_1" not in df.columns:
        df["aqi_lag_1"] = df["aqi"].shift(1)
        print("Added aqi_lag_1 feature")
    
    if "aqi_lag_3" not in df.columns:
        df["aqi_lag_3"] = df["aqi"].shift(3)
        print("Added aqi_lag_3 feature")
    
    if "aqi_lag_6" not in df.columns:
        df["aqi_lag_6"] = df["aqi"].shift(6)
        print("Added aqi_lag_6 feature")
    
    if "aqi_rolling_mean" not in df.columns:
        df["aqi_rolling_mean"] = df["aqi"].rolling(window=6, min_periods=1).mean()
        print("Added aqi_rolling_mean feature")
    
    if "aqi_rolling_std" not in df.columns:
        df["aqi_rolling_std"] = df["aqi"].rolling(window=6, min_periods=1).std().fillna(0)
        print("Added aqi_rolling_std feature")
    
    if "aqi_change" not in df.columns:
        df["aqi_change"] = df["aqi"].diff().fillna(0)
        print("Added aqi_change feature")
    
    # ============================================
    # STEP 7: Add pollutant ratios
    # ============================================
    print("\n" + "-"*40)
    print("STEP 7: Adding pollutant ratio features")
    print("-"*40)
    
    if "pm25_pm10_ratio" not in df.columns and "pm2_5" in df.columns and "pm10" in df.columns:
        df["pm25_pm10_ratio"] = df["pm2_5"] / (df["pm10"] + 1e-6)
        print("Added pm25_pm10_ratio feature")
    
    if "no2_so2_ratio" not in df.columns and "no2" in df.columns and "so2" in df.columns:
        df["no2_so2_ratio"] = df["no2"] / (df["so2"] + 1e-6)
        print("Added no2_so2_ratio feature")
    
    # ============================================
    # STEP 8: Add AQI class for classification
    # ============================================
    print("\n" + "-"*40)
    print("STEP 8: Adding AQI class labels")
    print("-"*40)
    
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
    
    if "aqi_class" not in df.columns:
        df["aqi_class"] = df["aqi"].apply(get_aqi_class)
        print("Added aqi_class feature")
    
    # ============================================
    # STEP 9: Drop rows with NaN from lag features
    # ============================================
    print("\n" + "-"*40)
    print("STEP 9: Removing rows with NaN values")
    print("-"*40)
    
    before = len(df)
    
    # Columns that shouldn't have NaN
    critical_cols = ["aqi", "pm2_5", "pm10", "hour", "day", "month", "aqi_lag_1", "aqi_rolling_mean"]
    existing_critical = [col for col in critical_cols if col in df.columns]
    
    df = df.dropna(subset=existing_critical)
    df = df.reset_index(drop=True)
    
    after = len(df)
    if before != after:
        print(f"Removed {before - after} rows with NaN values")
    
    # ============================================
    # STEP 10: Save cleaned data
    # ============================================
    print("\n" + "-"*40)
    print("STEP 10: Saving cleaned data")
    print("-"*40)
    
    # Save to multiple locations for compatibility
    output_paths = [
        "data/processed_aqi_data.csv",
        "data/cleaned_aqi_data.csv",
        "data/aqi_data_cleaned.csv"
    ]
    
    for output_path in output_paths:
        df.to_csv(output_path, index=False)
        print(f"Saved to: {output_path}")
    
    # Also backup original if it exists
    if loaded_path and loaded_path != "data/processed_aqi_data.csv":
        backup_path = loaded_path.replace(".csv", "_backup.csv")
        if not os.path.exists(backup_path):
            original_df = pd.read_csv(loaded_path)
            original_df.to_csv(backup_path, index=False)
            print(f"Original data backed up to: {backup_path}")
    
    # ============================================
    # FINAL SUMMARY
    # ============================================
    print("\n" + "="*60)
    print("DATA CLEANING COMPLETED")
    print("="*60)
    
    print(f"\nFinal shape: {df.shape}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
    
    print("\nColumn list:")
    for col in df.columns:
        print(f"  - {col}")
    
    print("\nAQI Class Distribution:")
    if "aqi_class" in df.columns:
        class_counts = df["aqi_class"].value_counts().sort_index()
        for cls, count in class_counts.items():
            class_name = ["Good", "Moderate", "Unhealthy Sensitive", "Unhealthy", "Very Unhealthy", "Hazardous"][cls-1]
            print(f"  Class {cls} ({class_name}): {count} samples ({count/len(df)*100:.1f}%)")
    
    print("\nData Statistics:")
    print(f"  Mean AQI: {df['aqi'].mean():.2f}")
    print(f"  Min AQI: {df['aqi'].min()}")
    print(f"  Max AQI: {df['aqi'].max()}")
    print(f"  Mean PM2.5: {df['pm2_5'].mean():.2f}")
    print(f"  Mean PM10: {df['pm10'].mean():.2f}")
    
    # Check if data is ready for training
    required_cols = ["aqi", "pm2_5", "hour", "day", "month", "aqi_lag_1", "aqi_rolling_mean"]
    missing_req = [col for col in required_cols if col not in df.columns]
    
    if missing_req:
        print(f"\nWarning: Missing required columns: {missing_req}")
        print("Training may fail. Please check the data.")
    else:
        print("\nData is ready for training!")
        print("Run: python pipelines/training_pipeline.py")
    
    print("\n" + "="*60)
    
    return True

def verify_cleaned_data():
    """Verify that cleaned data is properly formatted"""
    print("\n" + "="*60)
    print("VERIFYING CLEANED DATA")
    print("="*60)
    
    file_path = "data/processed_aqi_data.csv"
    
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found!")
        return False
    
    df = pd.read_csv(file_path)
    
    print(f"\nShape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")
    
    print("\nData types:")
    print(df.dtypes)
    
    print("\nMissing values:")
    print(df.isnull().sum())
    
    print("\nFirst 5 rows:")
    print(df.head())
    
    print("\nLast 5 rows:")
    print(df.tail())
    
    print("\nBasic statistics:")
    print(df.describe())
    
    return True

if __name__ == "__main__":
    success = clean_existing_csv()
    if success:
        verify_cleaned_data()