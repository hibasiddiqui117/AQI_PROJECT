import requests
import pandas as pd
from datetime import datetime
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

API_KEY = os.getenv("OPENWEATHER_API_KEY")
LAT = os.getenv("LAT")
LON = os.getenv("LON")

# OpenWeather Air Pollution API
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

    df = pd.DataFrame([row])

    os.makedirs("data", exist_ok=True)

    file_path = "data/aqi_data.csv"

    # Append data if file exists
    if os.path.exists(file_path):
        df.to_csv(file_path, mode='a', header=False, index=False)
    else:
        df.to_csv(file_path, index=False)

    print("✅ AQI Data Saved Successfully!")
    print(df)

else:
    print("❌ Failed to fetch data")
    print(response.text)