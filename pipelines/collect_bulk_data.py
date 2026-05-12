import requests
import pandas as pd
from datetime import datetime
import os
import time
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OPENWEATHER_API_KEY")
LAT = os.getenv("LAT")
LON = os.getenv("LON")

file_path = "data/aqi_data.csv"

os.makedirs("data", exist_ok=True)

print("🚀 Starting Bulk AQI Collection...")

for i in range(100):

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

        if os.path.exists(file_path):
            df.to_csv(file_path, mode="a", header=False, index=False)
        else:
            df.to_csv(file_path, index=False)

        print(f"✅ Row {i+1} collected")

    else:
        print("❌ API Failed")

    # wait 5 seconds
    time.sleep(5)

print("Bulk Data Collection Completed!")