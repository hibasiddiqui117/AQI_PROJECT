import streamlit as st
import requests
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="AQI Dashboard", layout="wide")

st.title("🌍 AI-Powered AQI Prediction Dashboard")
st.write("Real-time Air Quality Prediction System")

# -----------------------------
# SESSION STATE (HISTORY)
# -----------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# -----------------------------
# INPUT UI
# -----------------------------
col1, col2, col3 = st.columns(3)

with col1:
    co = st.number_input("CO", value=95.0)
    no = st.number_input("NO", value=0.01)
    no2 = st.number_input("NO2", value=0.06)
    o3 = st.number_input("O3", value=67.0)

with col2:
    so2 = st.number_input("SO2", value=0.3)
    pm2_5 = st.number_input("PM2.5", value=23.0)
    pm10 = st.number_input("PM10", value=82.0)
    nh3 = st.number_input("NH3", value=0.0)

with col3:
    hour = st.number_input("Hour", value=17)
    day = st.number_input("Day", value=12)
    month = st.number_input("Month", value=5)
    aqi_lag_1 = st.number_input("AQI Lag 1", value=3.0)
    aqi_rolling_mean = st.number_input("Rolling Mean", value=3.0)
    aqi_change = st.number_input("AQI Change", value=0.0)

# -----------------------------
# PREDICTION FUNCTION
# -----------------------------
def get_aqi_category(pm):
    if pm <= 50:
        return "Good", "🟢"
    elif pm <= 100:
        return "Moderate", "🟡"
    elif pm <= 150:
        return "Unhealthy", "🟠"
    else:
        return "Hazardous", "🔴"

# -----------------------------
# PREDICT BUTTON
# -----------------------------
if st.button("🔍 Predict AQI"):

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

    response = requests.post("http://127.0.0.1:5000/predict", json=payload)
    result = response.json()

    if "predicted_pm2_5" in result:

        prediction = result["predicted_pm2_5"]

        category, emoji = get_aqi_category(prediction)

        # -----------------------------
        # RESULT DISPLAY (CARDS)
        # -----------------------------
        st.markdown("## 📊 Prediction Result")

        colA, colB, colC = st.columns(3)

        with colA:
            st.metric("Predicted PM2.5", f"{prediction:.2f}")

        with colB:
            st.metric("AQI Category", category)

        with colC:
            st.metric("Status", emoji)

        # -----------------------------
        # ALERT SYSTEM
        # -----------------------------
        if category == "Good":
            st.success("Air quality is healthy ✅")
        elif category == "Moderate":
            st.warning("Air quality is acceptable but be cautious ⚠️")
        elif category == "Unhealthy":
            st.error("Unhealthy air! Limit outdoor activity 🚨")
        else:
            st.error("Hazardous air! Stay indoors immediately ⛔")

        # -----------------------------
        # SAVE HISTORY
        # -----------------------------
        st.session_state.history.append({
            "pm2_5": prediction,
            "category": category
        })

    else:
        st.error(result.get("error", "Prediction failed"))

# -----------------------------
# HISTORY CHART
# -----------------------------
if len(st.session_state.history) > 1:

    st.markdown("## 📈 Prediction History")

    df = pd.DataFrame(st.session_state.history)

    fig = px.line(
        df,
        y="pm2_5",
        title="PM2.5 Trend Over Time"
    )

    st.plotly_chart(fig, use_container_width=True)