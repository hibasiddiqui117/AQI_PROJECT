import streamlit as st
import requests

st.title("🌍 AQI Prediction Dashboard")

st.write("Enter pollutant values:")

co = st.number_input("CO")
no = st.number_input("NO")
no2 = st.number_input("NO2")
o3 = st.number_input("O3")
so2 = st.number_input("SO2")
pm2_5 = st.number_input("PM2.5")
pm10 = st.number_input("PM10")
nh3 = st.number_input("NH3")

hour = st.number_input("Hour")
day = st.number_input("Day")
month = st.number_input("Month")

aqi_lag_1 = st.number_input("AQI Lag 1")
aqi_rolling_mean = st.number_input("Rolling Mean")
aqi_change = st.number_input("AQI Change")

if st.button("Predict AQI"):

    payload = {
        "co": co,
        "no": no,
        "no2": no2,
        "o3": o3,
        "so2": so2,
        "pm2_5": pm2_5,
        "pm10": pm10,
        "nh3": nh3,
        "hour": hour,
        "day": day,
        "month": month,
        "aqi_lag_1": aqi_lag_1,
        "aqi_rolling_mean": aqi_rolling_mean,
        "aqi_change": aqi_change
    }

    try:
        response = requests.post("http://127.0.0.1:5000/predict", json=payload)
        result = response.json()

        st.success(f"Predicted PM2.5: {result['predicted_pm2_5']}")

    except Exception as e:
        st.error(f"Error connecting to API: {e}")