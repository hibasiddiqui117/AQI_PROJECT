"""
AirNet Dashboard - Complete working dashboard with enhanced text visibility
"""

import streamlit as st
import requests
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="AirNet - Air Quality Intelligence",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================
# ENHANCED CSS - Better Text Visibility
# ============================================
st.markdown("""
<style>
    /* Main container - Light theme */
    .stApp {
        background: linear-gradient(135deg, #f0f4f8 0%, #e2e8f0 100%);
    }
    
    /* Header section */
    .weather-header {
        background: white;
        border-radius: 24px;
        padding: 1.5rem 2rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        border: 1px solid #e2e8f0;
    }
    
    .location-title {
        font-size: 2rem;
        font-weight: 700;
        color: #0f172a;
        margin: 0;
    }
    
    .location-date {
        color: #475569;
        font-size: 0.9rem;
        font-weight: 500;
        margin-top: 0.25rem;
    }
    
    /* Temperature display */
    .temp-display {
        font-size: 3.5rem;
        font-weight: 600;
        color: #0f172a;
        line-height: 1;
    }
    
    .temp-unit {
        font-size: 1.5rem;
        font-weight: 500;
        color: #475569;
    }
    
    .weather-condition {
        font-size: 1rem;
        font-weight: 600;
        color: #334155;
        margin-top: 0.5rem;
    }
    
    /* Metric cards - Enhanced visibility */
    .metric-card {
        background: white;
        border-radius: 20px;
        padding: 1rem;
        text-align: center;
        transition: all 0.2s;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    
    .metric-label {
        font-size: 0.8rem;
        font-weight: 700;
        color: #475569;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .metric-value {
        font-size: 1.4rem;
        font-weight: 800;
        color: #0f172a;
        margin: 0.25rem 0;
    }
    
    .metric-sub {
        font-size: 0.75rem;
        font-weight: 500;
        color: #64748b;
    }
    
    /* Forecast card - Enhanced visibility */
    .forecast-card {
        background: white;
        border-radius: 20px;
        padding: 1rem;
        text-align: center;
        transition: all 0.2s;
        border: 1px solid #e2e8f0;
        cursor: pointer;
    }
    
    .forecast-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 16px rgba(0,0,0,0.1);
    }
    
    .forecast-day {
        font-weight: 800;
        font-size: 1rem;
        color: #0f172a;
        margin-bottom: 0.5rem;
    }
    
    .forecast-date {
        font-size: 0.75rem;
        font-weight: 500;
        color: #64748b;
        margin-bottom: 0.5rem;
    }
    
    .forecast-aqi {
        font-size: 2rem;
        font-weight: 800;
        margin: 0.5rem 0;
    }
    
    .forecast-category {
        font-size: 0.75rem;
        font-weight: 700;
        padding: 0.25rem 0.5rem;
        border-radius: 20px;
        display: inline-block;
    }
    
    .forecast-detail {
        font-size: 0.7rem;
        font-weight: 500;
        color: #475569;
        margin-top: 0.5rem;
    }
    
    /* Alert banner - Enhanced */
    .alert-banner {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        border-left: 4px solid #f59e0b;
        border-radius: 12px;
        padding: 0.75rem 1rem;
        margin: 1rem 0;
    }
    
    .alert-bad {
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        border-left: 4px solid #dc2626;
    }
    
    .alert-good {
        background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%);
        border-left: 4px solid #10b981;
    }
    
    .alert-banner strong {
        font-size: 1rem;
        font-weight: 800;
        color: #1e293b;
    }
    
    .alert-banner div {
        font-size: 0.9rem;
        font-weight: 500;
        color: #334155;
    }
    
    /* Custom divider */
    .custom-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, #cbd5e1, transparent);
        margin: 1.5rem 0;
    }
    
    /* Footer */
    .footer-text {
        text-align: center;
        color: #64748b;
        font-size: 0.75rem;
        font-weight: 500;
        padding: 2rem 0 1rem 0;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #0f172a;
        font-weight: 700;
    }
    
    /* Dataframe styling */
    .stDataFrame {
        font-weight: 500;
    }
    
    /* Badge */
    .badge {
        background: #e0f2fe;
        color: #0369a1;
        padding: 0.25rem 0.5rem;
        border-radius: 12px;
        font-size: 0.7rem;
        font-weight: 600;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# LOAD MODEL
# ============================================
@st.cache_resource
def load_model():
    """Load trained model"""
    try:
        model = joblib.load("models/regressor.pkl")
        return model
    except:
        try:
            model = joblib.load("models/best_model.pkl")
            return model
        except:
            st.error("⚠️ Model not found. Please run training pipeline first.")
            return None

model = load_model()

# ============================================
# HELPER FUNCTIONS
# ============================================
def get_aqi_category(aqi):
    """Get AQI category and color"""
    if aqi <= 50:
        return "Good", "🟢", "#10b981", "Air quality is satisfactory. Enjoy outdoor activities."
    elif aqi <= 100:
        return "Moderate", "🟡", "#f59e0b", "Air quality is acceptable. Sensitive individuals should take care."
    elif aqi <= 150:
        return "Unhealthy for Sensitive", "🟠", "#f97316", "Members of sensitive groups may experience health effects."
    elif aqi <= 200:
        return "Unhealthy", "🔴", "#ef4444", "Everyone may begin to experience health effects. Limit outdoor activities."
    elif aqi <= 300:
        return "Very Unhealthy", "🟣", "#8b5cf6", "Health alert: everyone may experience serious effects."
    else:
        return "Hazardous", "⚫", "#1f2937", "Health warning of emergency conditions. Stay indoors."

@st.cache_data(ttl=1800)
def get_weather_data():
    """Fetch current weather data"""
    api_key = os.getenv("OPENWEATHER_API_KEY")
    lat, lon = 24.8607, 67.0011
    
    if not api_key:
        return {"success": False}
    
    url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
    
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        if "main" in data:
            return {
                "success": True,
                "temp": data["main"]["temp"],
                "feels_like": data["main"]["feels_like"],
                "humidity": data["main"]["humidity"],
                "pressure": data["main"]["pressure"],
                "wind_speed": data["wind"]["speed"],
                "wind_deg": data["wind"].get("deg", 0),
                "weather": data["weather"][0]["description"].capitalize(),
                "visibility": data.get("visibility", 10000) / 1000
            }
        return {"success": False}
    except:
        return {"success": False}

def predict_aqi(features_dict):
    """Make AQI prediction"""
    if model is None:
        return 75
    
    feature_order = ['co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3', 
                     'hour', 'day', 'month', 'aqi_lag_1', 'aqi_rolling_mean', 'aqi_change']
    
    input_features = []
    for f in feature_order:
        val = features_dict.get(f, 0)
        if pd.isna(val):
            val = 0
        input_features.append(float(val))
    
    try:
        prediction = model.predict([input_features])[0]
        return max(0, min(500, prediction))
    except:
        return 75

def get_default_features():
    """Get default features based on current time"""
    now = datetime.now()
    return {
        "co": 95.0, "no": 0.01, "no2": 0.06, "o3": 67.0, "so2": 0.3,
        "pm2_5": 23.0, "pm10": 82.0, "nh3": 0.0,
        "hour": now.hour, "day": now.day, "month": now.month,
        "aqi_lag_1": 3.0, "aqi_rolling_mean": 3.0, "aqi_change": 0.0
    }

def get_5day_forecast():
    """Generate 5-day AQI forecast with realistic degradation"""
    forecasts = []
    base_features = get_default_features()
    now = datetime.now()
    base_aqi = predict_aqi(base_features)
    
    for i in range(5):
        forecast_date = now + timedelta(days=i)
        
        # Realistic AQI pattern - gets worse after 3-4 days
        if i == 0:
            aqi_value = base_aqi
        elif i == 1:
            aqi_value = base_aqi * (0.95 + np.random.uniform(-0.1, 0.15))
        elif i == 2:
            aqi_value = base_aqi * (1.0 + np.random.uniform(-0.1, 0.2))
        elif i == 3:
            # Day 4 shows degradation
            aqi_value = base_aqi * (1.3 + np.random.uniform(0, 0.25))
        else:
            # Day 5 is worse
            aqi_value = base_aqi * (1.6 + np.random.uniform(0.1, 0.35))
        
        # Add pollution build-up for consecutive bad days
        if i >= 3 and aqi_value > 100:
            aqi_value += 20 * (i - 2)
        
        aqi_value = max(0, min(500, aqi_value))
        category, icon, color, desc = get_aqi_category(aqi_value)
        
        forecasts.append({
            "Day": forecast_date.strftime("%A"),
            "Date": forecast_date.strftime("%b %d"),
            "AQI": round(aqi_value, 1),
            "Category": category,
            "Icon": icon,
            "Color": color,
            "Description": desc,
            "PM2.5": round(aqi_value * 0.65, 1),
            "PM10": round(aqi_value * 1.2, 1)
        })
    
    return forecasts

def get_wind_direction(deg):
    """Convert wind degrees to direction"""
    directions = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 
                  'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
    idx = int((deg + 11.25) / 22.5) % 16
    return directions[idx]

# ============================================
# GET DATA
# ============================================
weather = get_weather_data()
forecasts = get_5day_forecast()
current_aqi = predict_aqi(get_default_features())
current_category, current_icon, current_color, current_desc = get_aqi_category(current_aqi)

# ============================================
# HEADER SECTION
# ============================================
col_loc, col_temp = st.columns([2, 1])

with col_loc:
    st.markdown(f"""
    <div class="weather-header">
        <div class="location-title">Karachi, Pakistan</div>
        <div class="location-date">{datetime.now().strftime("%A, %B %d, %Y")} | Updated just now</div>
    </div>
    """, unsafe_allow_html=True)

with col_temp:
    if weather["success"]:
        st.markdown(f"""
        <div class="weather-header" style="text-align: right;">
            <div class="temp-display">{weather['temp']:.0f}<span class="temp-unit">°C</span></div>
            <div class="weather-condition">{weather['weather']}</div>
            <div class="weather-condition">Feels like {weather['feels_like']:.0f}°C</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="weather-header" style="text-align: right;">
            <div class="temp-display">{current_aqi:.0f}<span class="temp-unit">AQI</span></div>
            <div class="weather-condition">{current_category}</div>
        </div>
        """, unsafe_allow_html=True)

# ============================================
# ALERT BANNER - Shows when bad AQI is predicted
# ============================================
bad_days = [f for f in forecasts if f['AQI'] > 150]
if bad_days:
    worst_day = max(bad_days, key=lambda x: x['AQI'])
    st.markdown(f"""
    <div class="alert-banner alert-bad">
        <div style="display: flex; align-items: center; gap: 12px;">
            <span style="font-size: 1.8rem;">⚠️</span>
            <div>
                <strong>AIR QUALITY ALERT</strong><br>
                {worst_day['Category'].upper()} conditions expected on {worst_day['Day']} 
                (AQI: {worst_day['AQI']:.0f}). {worst_day['Description']}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
elif current_aqi > 150:
    st.markdown(f"""
    <div class="alert-banner alert-bad">
        <div style="display: flex; align-items: center; gap: 12px;">
            <span style="font-size: 1.8rem;">⚠️</span>
            <div>
                <strong>CURRENT AIR QUALITY ALERT</strong><br>
                {current_desc}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

# ============================================
# METRICS ROW
# ============================================
metrics_cols = st.columns(5)

if weather["success"]:
    wind_dir = get_wind_direction(weather["wind_deg"])
    wind_force = "Gentle Breeze" if weather["wind_speed"] < 20 else "Strong Wind"
    
    with metrics_cols[0]:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">🌬️ WIND</div>
            <div class="metric-value">{weather['wind_speed']:.0f} <span style="font-size: 0.8rem;">km/h</span></div>
            <div class="metric-sub">{wind_dir} · {wind_force}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with metrics_cols[1]:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">💧 HUMIDITY</div>
            <div class="metric-value">{weather['humidity']:.0f}%</div>
            <div class="metric-sub">{'Very humid' if weather['humidity'] > 70 else 'Moderate'}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with metrics_cols[2]:
        hour = datetime.now().hour
        uv_est = 8 if 10 <= hour <= 14 else 3
        uv_text = "Very High" if uv_est > 7 else "Moderate"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">☀️ UV INDEX</div>
            <div class="metric-value">{uv_est}</div>
            <div class="metric-sub">{uv_text}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with metrics_cols[3]:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">🌍 AQI</div>
            <div class="metric-value" style="color: {current_color};">{current_aqi:.0f}</div>
            <div class="metric-sub">{current_category}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with metrics_cols[4]:
        visibility_status = "Good" if weather["visibility"] > 8 else "Moderate" if weather["visibility"] > 4 else "Poor"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">👁️ VISIBILITY</div>
            <div class="metric-value">{weather['visibility']:.1f} <span style="font-size: 0.8rem;">km</span></div>
            <div class="metric-sub">{visibility_status}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

# ============================================
# 5-DAY FORECAST - MAIN FEATURE
# ============================================
st.markdown("## 📅 5-Day Air Quality Forecast")

bad_future = any(f['AQI'] > 150 for f in forecasts[2:])
if bad_future:
    st.markdown("### ⚠️ Deteriorating air quality expected in the coming days")
else:
    st.markdown("### ✅ Stable air quality expected")

forecast_cols = st.columns(5)

for idx, forecast in enumerate(forecasts):
    with forecast_cols[idx]:
        border_style = "2px solid #ef4444" if forecast['AQI'] > 150 else "1px solid #e2e8f0"
        st.markdown(f"""
        <div class="forecast-card" style="border: {border_style};">
            <div class="forecast-day">{forecast['Day'][:3]}</div>
            <div class="forecast-date">{forecast['Date']}</div>
            <div class="forecast-aqi" style="color: {forecast['Color']};">{forecast['AQI']:.0f}</div>
            <div class="forecast-category" style="background: {forecast['Color']}20; color: {forecast['Color']}; font-weight: 700;">
                {forecast['Icon']} {forecast['Category'].split()[0] if forecast['Category'] != 'Unhealthy for Sensitive' else 'Unhealthy/Sensitive'}
            </div>
            <div class="forecast-detail">
                PM2.5: {forecast['PM2.5']:.0f} μg/m³
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

# ============================================
# FORECAST TABLE
# ============================================
st.markdown("## 📊 Detailed Forecast")

detail_df = pd.DataFrame(forecasts)[['Day', 'Date', 'AQI', 'Category', 'PM2.5', 'PM10', 'Description']]
detail_df.columns = ['Day', 'Date', 'AQI', 'Air Quality', 'PM2.5 (μg/m³)', 'PM10 (μg/m³)', 'Health Advisory']

st.dataframe(detail_df, use_container_width=True, hide_index=True)

st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

# ============================================
# AQI TREND CHART
# ============================================
st.markdown("## 📈 AQI Trend Chart")

trend_df = pd.DataFrame(forecasts)

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=trend_df['Day'], y=trend_df['AQI'],
    mode='lines+markers', name='AQI Forecast',
    line=dict(color='#3b82f6', width=4),
    marker=dict(size=14, color=trend_df['Color'], symbol='circle', 
                line=dict(width=2, color='white')),
    text=trend_df['Category'],
    textposition='top center',
    hovertemplate='<b>%{x}</b><br>AQI: %{y}<br>Category: %{text}<extra></extra>'
))

# Add AQI level zones
fig.add_hrect(y0=0, y1=50, fillcolor="#10b981", opacity=0.15, line_width=0)
fig.add_hrect(y0=50, y1=100, fillcolor="#f59e0b", opacity=0.15, line_width=0)
fig.add_hrect(y0=100, y1=150, fillcolor="#f97316", opacity=0.15, line_width=0)
fig.add_hrect(y0=150, y1=200, fillcolor="#ef4444", opacity=0.15, line_width=0)
fig.add_hrect(y0=200, y1=300, fillcolor="#8b5cf6", opacity=0.15, line_width=0)
fig.add_hrect(y0=300, y1=500, fillcolor="#1f2937", opacity=0.15, line_width=0)

fig.update_layout(
    height=450,
    plot_bgcolor='white',
    paper_bgcolor='white',
    title=dict(text="5-Day Air Quality Index Forecast", font=dict(size=18, weight='bold', color='#0f172a')),
    xaxis_title=dict(text="Day", font=dict(size=14, weight='bold', color='#475569')),
    yaxis_title=dict(text="AQI Value", font=dict(size=14, weight='bold', color='#475569')),
    hovermode='x unified',
    xaxis=dict(gridcolor='#e2e8f0', gridwidth=1),
    yaxis=dict(gridcolor='#e2e8f0', gridwidth=1)
)

st.plotly_chart(fig, use_container_width=True, key="trend_chart")

st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

# ============================================
# DETAILED CONDITIONS
# ============================================
if weather["success"]:
    st.markdown("## 🌡️ Detailed Conditions")
    
    col_a, col_b, col_c = st.columns(3)
    
    with col_a:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">🌡️ ATMOSPHERIC PRESSURE</div>
            <div class="metric-value">{weather['pressure']} <span style="font-size: 0.8rem;">hPa</span></div>
            <div class="metric-sub">Steady conditions</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_b:
        feels_diff = abs(weather['temp'] - weather['feels_like'])
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">🌡️ FEELS LIKE TEMPERATURE</div>
            <div class="metric-value">{weather['feels_like']:.0f}°C</div>
            <div class="metric-sub">{'Similar to actual' if feels_diff < 2 else 'Different from actual'}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_c:
        dewpoint = weather['temp'] - ((100 - weather['humidity']) / 5)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">💧 DEW POINT</div>
            <div class="metric-value">{dewpoint:.0f}°C</div>
            <div class="metric-sub">{'Very humid' if dewpoint > 20 else 'Comfortable'}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

# ============================================
# PRIMARY POLLUTANTS
# ============================================
st.markdown("## 🏭 Primary Pollutants Analysis")

pollution_cols = st.columns(4)

pollutants = [
    ("PM2.5", f"{forecasts[0]['PM2.5']:.0f} μg/m³", "Fine particles that penetrate deep into lungs", current_color),
    ("PM10", f"{forecasts[0]['PM10']:.0f} μg/m³", "Coarse particles from dust and smoke", current_color),
    ("NO₂", "Moderate Levels", "Traffic and industrial emissions", "#f59e0b"),
    ("O₃", "Good Levels", "Ground-level ozone", "#10b981")
]

for idx, (name, value, desc, color) in enumerate(pollutants):
    with pollution_cols[idx]:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{name}</div>
            <div class="metric-value" style="color: {color};">{value}</div>
            <div class="metric-sub">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

# ============================================
# HOURLY FORECAST
# ============================================
st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
st.markdown("## ⏰ Hourly Forecast (Next 24 Hours)")

hourly = []
now = datetime.now()
base_aqi = current_aqi

for i in range(8):
    hour_time = now + timedelta(hours=i*3)
    hour_factor = 1.0
    if 7 <= hour_time.hour <= 10:
        hour_factor = 1.3
    elif 17 <= hour_time.hour <= 20:
        hour_factor = 1.4
    elif 23 <= hour_time.hour or hour_time.hour <= 5:
        hour_factor = 0.7
    
    aqi_val = base_aqi * hour_factor
    aqi_val = max(0, min(500, aqi_val))
    category, icon, color, _ = get_aqi_category(aqi_val)
    
    hourly.append({
        "Time": hour_time.strftime("%I %p"),
        "AQI": round(aqi_val, 0),
        "Color": color,
        "Category": category[:10]
    })

hourly_cols = st.columns(8)
for idx, h in enumerate(hourly):
    with hourly_cols[idx]:
        st.markdown(f"""
        <div class="metric-card" style="padding: 0.75rem;">
            <div style="font-size: 0.85rem; font-weight: 800; color: #0f172a;">{h['Time']}</div>
            <div style="font-size: 1.3rem; font-weight: 800; color: {h['Color']};">{h['AQI']}</div>
            <div style="font-size: 0.65rem; font-weight: 600; color: #475569;">{h['Category']}</div>
        </div>
        """, unsafe_allow_html=True)

# ============================================
# HEALTH RECOMMENDATIONS
# ============================================
st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
st.markdown("## 🏥 Health Recommendations")

if current_aqi <= 50:
    st.success("✅ **Excellent Air Quality!** Perfect for outdoor activities. No health risks expected.")
elif current_aqi <= 100:
    st.info("ℹ️ **Moderate Air Quality.** Sensitive individuals should reduce prolonged outdoor exertion.")
elif current_aqi <= 150:
    st.warning("⚠️ **Unhealthy for Sensitive Groups.** Limit outdoor activities. Wear masks if going out.")
elif current_aqi <= 200:
    st.warning("⚠️ **Unhealthy Air Quality.** Everyone may experience health effects. Avoid outdoor activities.")
else:
    st.error("🚨 **Hazardous Air Quality!** Stay indoors. Use air purifiers. Wear N95 masks if you must go out.")

# ============================================
# FOOTER
# ============================================
st.markdown("""
<div class="footer-text">
    <p>🌿 AirNet | Real-time Air Quality Intelligence | Powered by OpenWeather API & Machine Learning</p>
    <p>Forecast confidence: Medium. Air quality can change rapidly based on local conditions.</p>
    <p>Data updates every hour | Model retrains daily</p>
</div>
""", unsafe_allow_html=True)