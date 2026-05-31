import requests
import pandas as pd
from datetime import datetime
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import traceback

from dotenv import load_dotenv
from src.feature_store.dagshub_feature_store import save_features

# Load env variables
load_dotenv()

print("="*60)
print("FEATURE PIPELINE STARTING")
print("="*60)

# Print debug info
print(f"\n Current working directory: {os.getcwd()}")
print(f"   Files in directory: {os.listdir('.')}")

# Check for API key
API_KEY = os.getenv("OPENWEATHER_API_KEY")
LAT = os.getenv("LAT", "24.8607")
LON = os.getenv("LON", "67.0011")

print(f"\n API Key present: {'Yes' if API_KEY else 'No'}")
print(f" Coordinates: Lat={LAT}, Lon={LON}")

# Validate API key
if not API_KEY:
    print("\n❌ ERROR: OPENWEATHER_API_KEY not found in environment!")
    print("Please make sure you've added the secret in GitHub Actions:")
    print("  Settings → Secrets and variables → Actions → Add secret")
    sys.exit(1)

# API URL
url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={LAT}&lon={LON}&appid={API_KEY}"

print(f"\n Fetching data from API...")
print(f"URL: {url.replace(API_KEY, 'HIDDEN')}")

try:
    response = requests.get(url, timeout=30)
    print(f"Response status code: {response.status_code}")
    
except requests.exceptions.RequestException as e:
    print(f"\n❌ Network error: {e}")
    traceback.print_exc()
    sys.exit(1)

if response.status_code == 200:
    print("\n API Request Successful!")
    
    try:
        data = response.json()
        pollution = data["list"][0]
        
        print(f"\n Raw API Response:")
        print(f"  - AQI: {pollution['main']['aqi']}")
        print(f"  - PM2.5: {pollution['components']['pm2_5']}")
        print(f"  - PM10: {pollution['components']['pm10']}")
        
        row = {
            "datetime": datetime.now(),
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
        
        df_new = pd.DataFrame([row])
        print(f"\n Created new data row")
        
        # Create data directory
        os.makedirs("data", exist_ok=True)
        print(f" Created/verified data directory")
        
        file_path = "data/aqi_data.csv"
        
        # If file exists, append safely
        if os.path.exists(file_path):
            print(f" Existing file found at {file_path}")
            try:
                df_existing = pd.read_csv(file_path)
                print(f"  - Loaded {len(df_existing)} existing records")
                
                # Check if datetime column exists
                if 'datetime' not in df_existing.columns:
                    print(f"  'datetime' column not found, recreating file")
                    df_final = df_new
                else:
                    df_final = pd.concat([df_existing, df_new], ignore_index=True)
                    # Remove duplicates
                    df_final = df_final.drop_duplicates(subset=["datetime"], keep='last')
                    print(f"  - Combined: {len(df_final)} total records")
            except Exception as e:
                print(f"  Error reading existing file: {e}")
                print(f"  Creating new file instead")
                df_final = df_new
        else:
            print(f" No existing file found, creating new one")
            df_final = df_new
        
        # Save updated dataset
        try:
            print("\n Saving features to DagsHub Feature Store...")

            save_features(df_final, "latest_features")

            print(" Feature store update successful")

        except Exception as e:
            print(f" Feature store error (non-critical): {e}")

        # -----------------------------
        # SAVE LOCALLY

        df_final.to_csv(file_path, index=False)
        print(f"\n Data saved to {file_path}")
        print(f" Total records in file: {len(df_final)}")

        # Verify save worked
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            print(f" File size: {file_size} bytes")
            if file_size > 0:
                print("\n Feature Pipeline Completed Successfully!")
                print("="*60)
                sys.exit(0)
            else:
                print("\n File saved but is empty!")
                sys.exit(1)
        else:
            print("\n File was not saved properly!")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n Error processing data: {e}")
        traceback.print_exc()
        sys.exit(1)

else:
    print(f"\n API Request Failed")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text}")
    sys.exit(1)