import requests
import pandas as pd
from datetime import datetime
import os
from dotenv import load_dotenv
import sys

# Load env variables
load_dotenv()

API_KEY = os.getenv("OPENWEATHER_API_KEY")
LAT = os.getenv("LAT", "24.8607")
LON = os.getenv("LON", "67.0011")

def fetch_aqi_data():
    """Fetch AQI data from OpenWeather API"""
    url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={LAT}&lon={LON}&appid={API_KEY}"
    
    print(f"🌐 Fetching data from API...")
    response = requests.get(url)
    
    if response.status_code != 200:
        print(f"❌ API Request Failed: {response.status_code}")
        print(response.text)
        return None
    
    data = response.json()
    pollution = data["list"][0]
    
    row = {
        "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),  # Convert to string for consistency
        "aqi": pollution["main"]["aqi"],
        "co": pollution["components"]["co"],
        "no": pollution["components"]["no"],
        "no2": pollution["components"]["no2"],
        "o3": pollution["components"]["o3"],
        "so2": pollution["components"]["so2"],
        "pm2_5": pollution["components"]["pm2_5"],
        "pm10": pollution["components"]["pm10"],
        "nh3": pollution["components"]["nh3"]
    }
    
    return pd.DataFrame([row])

def load_existing_data(file_path):
    """Load existing data safely"""
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        try:
            df_existing = pd.read_csv(file_path)
            # Convert datetime column if it exists
            if 'datetime' in df_existing.columns:
                df_existing['datetime'] = pd.to_datetime(df_existing['datetime'])
            print(f"📂 Loaded {len(df_existing)} existing records")
            return df_existing
        except Exception as e:
            print(f"⚠️ Error reading existing file: {e}")
            return pd.DataFrame()
    else:
        print(f"📁 No existing data found at {file_path}")
        return pd.DataFrame()

def main():
    print("🚀 Feature pipeline started...")
    
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    file_path = "data/aqi_data.csv"
    
    # Fetch new data
    df_new = fetch_aqi_data()
    if df_new is None:
        print("❌ Pipeline failed - no data fetched")
        sys.exit(1)
    
    print(f"📊 Fetched new data: AQI={df_new.iloc[0]['aqi']}, PM2.5={df_new.iloc[0]['pm2_5']:.1f}")
    
    # Load existing data
    df_existing = load_existing_data(file_path)
    
    # Combine data
    if len(df_existing) > 0:
        # Convert datetime for comparison
        df_new['datetime'] = pd.to_datetime(df_new['datetime'])
        
        # Combine and remove duplicates (keep only new unique timestamps)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined = df_combined.drop_duplicates(subset=['datetime'], keep='last')
        df_combined = df_combined.sort_values('datetime')
        
        print(f"✅ Combined {len(df_existing)} existing + 1 new = {len(df_combined)} total records")
    else:
        df_combined = df_new
        print(f"✅ Created new dataset with 1 record")
    
    # Save to CSV
    df_combined.to_csv(file_path, index=False)
    
    # Print summary
    print("\n" + "="*50)
    print("📈 DATASET SUMMARY")
    print("="*50)
    print(f"Total records: {len(df_combined)}")
    print(f"Date range: {df_combined['datetime'].min()} to {df_combined['datetime'].max()}")
    print(f"Latest AQI: {df_combined.iloc[-1]['aqi']}")
    print("\n✅ Feature Pipeline Updated Successfully!")
    
    # Show last 5 records
    print("\n📋 Last 5 records:")
    print(df_combined.tail())
    
    return df_combined

if __name__ == "__main__":
    main()