from flask import Flask, jsonify
import pandas as pd
import numpy as np
import requests
import datetime
import joblib
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# ------------------------------------------------
# LOAD MODEL
# ------------------------------------------------
model = joblib.load("models/regressor.pkl")

# ------------------------------------------------
# OPENWEATHER CONFIG
# ------------------------------------------------
API_KEY = os.getenv("OPENWEATHER_API_KEY")

LAT = 24.8607
LON = 67.0011

# ------------------------------------------------
# FETCH LIVE AQI DATA
# ------------------------------------------------
def fetch_live_data():

    url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={LAT}&lon={LON}&appid={API_KEY}"

    response = requests.get(url)

    data = response.json()

    pollution = data["list"][0]

    components = pollution["components"]

    current_time = datetime.datetime.now()

    feature_dict = {
        "co": components["co"],
        "no": components["no"],
        "no2": components["no2"],
        "o3": components["o3"],
        "so2": components["so2"],
        "pm2_5": components["pm2_5"],
        "pm10": components["pm10"],
        "nh3": components["nh3"],
        "hour": current_time.hour,
        "day": current_time.day,
        "month": current_time.month,
        "aqi_lag_1": pollution["main"]["aqi"],
        "aqi_rolling_mean": pollution["main"]["aqi"],
        "aqi_change": 0
    }

    return feature_dict

# ------------------------------------------------
# AQI CATEGORY
# ------------------------------------------------
def get_aqi_category(value):

    if value <= 50:
        return "Good"

    elif value <= 100:
        return "Moderate"

    elif value <= 150:
        return "Unhealthy"

    else:
        return "Hazardous"

# ------------------------------------------------
# HOME ROUTE
# ------------------------------------------------
@app.route("/")
def home():

    return "AQI Forecasting API Running"

# ------------------------------------------------
# 3 DAY FORECAST ROUTE
# ------------------------------------------------
@app.route("/forecast")
def forecast():

    try:

        base_features = fetch_live_data()

        predictions = []

        for i in range(1, 4):

            future_features = base_features.copy()

            future_features["day"] += i

            X = pd.DataFrame([future_features])

            prediction = model.predict(X)[0]

            predictions.append({
                "day": f"Day {i}",
                "predicted_aqi": round(float(prediction), 2),
                "category": get_aqi_category(prediction)
            })

        return jsonify(predictions)

    except Exception as e:

        return jsonify({
            "error": str(e)
        })

# ------------------------------------------------
# RUN APP
# ------------------------------------------------
if __name__ == "__main__":

    app.run(debug=True)
