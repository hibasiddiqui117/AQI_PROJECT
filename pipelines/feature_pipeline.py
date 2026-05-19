import requests
import pandas as pd
from datetime import datetime
import os
from dotenv import load_dotenv

# Load env variables
load_dotenv()

API_KEY = os.getenv("OPENWEATHER_API_KEY")
LAT = os.getenv("LAT", "24.8607")
LON = os.getenv("LON", "67.0011")

# API URL
url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={LAT}&lon={LON}&appid={API_KEY}"

response = requests.get(url)

if response.status_code == 200:

    data = response.json()

    pollution = data["list"][0]

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

    os.makedirs("data", exist_ok=True)

    file_path = "data/aqi_data.csv"

    # If file exists, append safely
    if os.path.exists(file_path):

        df_existing = pd.read_csv(file_path)

        df_final = pd.concat([df_existing, df_new], ignore_index=True)

        # Remove duplicates
        df_final.drop_duplicates(subset=["datetime"], inplace=True)

    else:
        df_final = df_new

    # Save updated dataset
    df_final.to_csv(file_path, index=False)
    print("🚀 Feature pipeline started...")
    print("Rows before save:", len(df_final))
    print("✅ Feature Pipeline Updated Successfully!")
    print(df_new)

else:
    print("❌ API Request Failed")
    print(response.text)