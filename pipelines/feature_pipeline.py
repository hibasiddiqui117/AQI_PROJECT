import requests
import pandas as pd
from datetime import datetime
import os
import sys
import traceback
from dotenv import load_dotenv

# -----------------------------
# PROJECT PATH FIX
# -----------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.feature_store.dagshub_feature_store import save_features, save_latest

# -----------------------------
# LOAD ENV
# -----------------------------
load_dotenv()

print("="*60)
print(" FEATURE PIPELINE STARTING")
print("="*60)

# -----------------------------
# DEBUG INFO
# -----------------------------
print(f"\n Current working directory: {os.getcwd()}")
print(f" Files in directory: {os.listdir('.')}")

# -----------------------------
# CONFIG
# -----------------------------
API_KEY = os.getenv("OPENWEATHER_API_KEY")
LAT = float(os.getenv("LAT", 24.8607))
LON = float(os.getenv("LON", 67.0011))

print(f"\n API Key present: {'Yes' if API_KEY else 'No'}")
print(f" Coordinates: Lat={LAT}, Lon={LON}")

# -----------------------------
# VALIDATE API KEY
# -----------------------------
if not API_KEY:
    print("\n ERROR: OPENWEATHER_API_KEY missing!")
    sys.exit(1)

# -----------------------------
# API CALL
# -----------------------------
url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={LAT}&lon={LON}&appid={API_KEY}"

print(f"\n Fetching data from API...")
print(f"URL: {url.replace(API_KEY, 'HIDDEN')}")

try:
    response = requests.get(url, timeout=30)
    print(f" Response status: {response.status_code}")

except Exception as e:
    print(f"\n Network error: {e}")
    traceback.print_exc()
    sys.exit(1)

# -----------------------------
# HANDLE RESPONSE
# -----------------------------
if response.status_code != 200:
    print("\n API FAILED")
    print(response.text)
    sys.exit(1)

try:
    data = response.json()
    pollution = data["list"][0]

    aqi = pollution["main"]["aqi"]
    components = pollution["components"]

    print("\n API SUCCESS")
    print(f"AQI: {aqi}")
    print(f"PM2.5: {components['pm2_5']}")
    print(f"PM10: {components['pm10']}")

    # -----------------------------
    # CREATE ROW
    # -----------------------------
    row = {
        "datetime": datetime.now(),
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

    # -----------------------------
    # DATA STORAGE
    # -----------------------------
    os.makedirs("data", exist_ok=True)
    file_path = "data/aqi_data.csv"

    if os.path.exists(file_path):
        df_existing = pd.read_csv(file_path)

        if "datetime" in df_existing.columns:
            df_final = pd.concat([df_existing, df_new], ignore_index=True)
            df_final.drop_duplicates(subset=["datetime"], inplace=True)
        else:
            df_final = df_new
    else:
        df_final = df_new

    # -----------------------------
    # SAVE RAW DATA
    # -----------------------------
    df_final.to_csv(file_path, index=False)

    print(f"\n Data saved locally: {len(df_final)} records")

    # -----------------------------
    # FEATURE STORE (DAGSHUB)
    # -----------------------------
    try:
        print("\n Saving to DagsHub Feature Store...")

        save_features(df_final, "latest_features")
        save_latest(df_final)

        print(" Feature store updated successfully")

    except Exception as e:
        print(f" Feature store warning (non-critical): {e}")

    print("\n FEATURE PIPELINE COMPLETED SUCCESSFULLY")
    print("="*60)

except Exception as e:
    print(f"\n Processing error: {e}")
    traceback.print_exc()
    sys.exit(1)