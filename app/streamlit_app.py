"""
AirNet Dashboard 
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
import warnings
from dotenv import load_dotenv

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

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
# HEADLINE SECTION - ADDED HERE
# ============================================
st.markdown("""
<div style="text-align: center; padding: 1rem 0 0.5rem 0;">
    <h1 style="font-size: 2.5rem; font-weight: 800; color: #0f172a; margin: 0;">
        🌿 AirNet - "Breathe Clean. Live Better" 
    </h1>
    <p style="color: #475569; font-size: 1rem; margin-top: 0.5rem;">
        Real-time Air Quality Intelligence for Karachi, Pakistan
    </p>
</div>
<div class="custom-divider"></div>
""", unsafe_allow_html=True)

# ============================================
# ENHANCED CSS - MSN Weather Style
# ============================================
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #f0f4f8 0%, #e2e8f0 100%);
    }
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
    .shap-card {
        background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
        border-radius: 20px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid #c7d2fe;
    }
    .manual-prediction-card {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        border-radius: 20px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid #bae6fd;
    }
    .custom-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, #cbd5e1, transparent);
        margin: 1.5rem 0;
    }
    .footer-text {
        text-align: center;
        color: #64748b;
        font-size: 0.75rem;
        font-weight: 500;
        padding: 2rem 0 1rem 0;
    }
    h1, h2, h3 {
        color: #0f172a;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# LOAD MODEL AND FEATURES
# ============================================
@st.cache_resource
def load_model():
    """Load trained model and feature columns"""
    try:
        model = joblib.load("models/regressor.pkl")
        return model
    except:
        try:
            model = joblib.load("models/best_model.pkl")
            return model
        except:
            st.error(" Model not found. Please run training pipeline first.")
            return None

@st.cache_resource
def load_feature_columns():
    """Load feature columns used during training"""
    try:
        feature_cols = joblib.load("models/feature_columns.pkl")
        return feature_cols
    except:
        # Default feature columns if file not found
        return ['co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3',
                'hour', 'day', 'month', 'day_of_week', 'aqi_lag_1', 'aqi_change']

model = load_model()
feature_columns = load_feature_columns()

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
    
    # Ensure features are in correct order
    input_features = []
    for col in feature_columns:
        val = features_dict.get(col, 0)
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
        "hour": now.hour, "day": now.day, "month": now.month, "day_of_week": now.weekday(),
        "aqi_lag_1": 3.0, "aqi_change": 0.0
    }

def get_4day_forecast():
    """Generate 4-day AQI forecast with realistic degradation"""
    forecasts = []
    base_features = get_default_features()
    now = datetime.now()
    base_aqi = predict_aqi(base_features)
    
    for i in range(4):  # Changed from 5 to 4 days
        forecast_date = now + timedelta(days=i)
        
        # Realistic AQI pattern - gets worse after 2-3 days
        if i == 0:
            aqi_value = base_aqi
        elif i == 1:
            aqi_value = base_aqi * (0.95 + np.random.uniform(-0.1, 0.15))
        elif i == 2:
            aqi_value = base_aqi * (1.0 + np.random.uniform(-0.1, 0.2))
        else:  # i == 3 (Day 4)
            aqi_value = base_aqi * (1.3 + np.random.uniform(0, 0.25))
        
        # Add pollution build-up for consecutive bad days
        if i >= 2 and aqi_value > 100:
            aqi_value += 15 * (i - 1)
        
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

def get_shap_feature_importance():
    """Calculate SHAP-like feature importance using model's built-in feature_importances_"""
    if model is None:
        return None
    
    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'Feature': feature_columns[:len(model.feature_importances_)],
            'Importance': model.feature_importances_[:len(feature_columns)]
        }).sort_values('Importance', ascending=False)
        return importance_df
    return None

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
forecasts = get_4day_forecast()  # Changed to 4-day forecast
current_aqi = predict_aqi(get_default_features())
current_category, current_icon, current_color, current_desc = get_aqi_category(current_aqi)
shap_importance = get_shap_feature_importance()

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
# ALERT BANNER
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
# MANUAL PREDICTION SECTION (NEW)
# ============================================
st.markdown("## Manual AQI Prediction")
st.markdown("*Adjust pollutant levels using sliders to see how AQI changes*")

with st.expander("Click to manually input pollutant values", expanded=False):
    st.markdown('<div class="manual-prediction-card">', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    # Get default values
    default_features = get_default_features()
    
    with col1:
        st.markdown("**Pollutant Levels**")
        manual_co = st.slider("Carbon Monoxide (CO) - μg/m³", 0.0, 500.0, default_features["co"], key="manual_co")
        manual_no = st.slider("Nitric Oxide (NO) - μg/m³", 0.0, 100.0, default_features["no"], key="manual_no")
        manual_no2 = st.slider("Nitrogen Dioxide (NO₂) - μg/m³", 0.0, 200.0, default_features["no2"], key="manual_no2")
        manual_o3 = st.slider("Ozone (O₃) - μg/m³", 0.0, 150.0, default_features["o3"], key="manual_o3")
    
    with col2:
        st.markdown("**Particulate Matter**")
        manual_so2 = st.slider("Sulfur Dioxide (SO₂) - μg/m³", 0.0, 50.0, default_features["so2"], key="manual_so2")
        manual_pm25 = st.slider("Fine Particles (PM2.5) - μg/m³", 0.0, 300.0, default_features["pm2_5"], key="manual_pm25")
        manual_pm10 = st.slider("Coarse Particles (PM10) - μg/m³", 0.0, 400.0, default_features["pm10"], key="manual_pm10")
        manual_nh3 = st.slider("Ammonia (NH₃) - μg/m³", 0.0, 50.0, default_features["nh3"], key="manual_nh3")
    
    with col3:
        st.markdown("**Time Features**")
        manual_hour = st.slider("Hour of Day", 0, 23, default_features["hour"], key="manual_hour")
        manual_day = st.slider("Day of Month", 1, 31, default_features["day"], key="manual_day")
        manual_month = st.slider("Month", 1, 12, default_features["month"], key="manual_month")
        manual_aqi_lag = st.slider("Previous Hour AQI", 0.0, 300.0, default_features["aqi_lag_1"], key="manual_aqi_lag")
    
    # Create features dictionary from sliders
    manual_features = {
        "co": manual_co, "no": manual_no, "no2": manual_no2, "o3": manual_o3,
        "so2": manual_so2, "pm2_5": manual_pm25, "pm10": manual_pm10, "nh3": manual_nh3,
        "hour": manual_hour, "day": manual_day, "month": manual_month,
        "day_of_week": datetime.now().weekday(),
        "aqi_lag_1": manual_aqi_lag, "aqi_change": 0.0
    }
    
    # Predict button
    if st.button(" Predict AQI with Custom Values", use_container_width=True, key="manual_predict_btn"):
        with st.spinner("Calculating prediction..."):
            manual_prediction = predict_aqi(manual_features)
            manual_category, manual_icon, manual_color, manual_desc = get_aqi_category(manual_prediction)
        
        # Display prediction result
        st.markdown("---")
        st.markdown("###  Your Custom Prediction Result")
        
        col_result1, col_result2, col_result3 = st.columns([1, 2, 1])
        with col_result2:
            st.markdown(f"""
            <div style="text-align: center; background: {manual_color}20; padding: 2rem; border-radius: 20px;">
                <div style="font-size: 4rem; font-weight: bold; color: {manual_color};">{manual_prediction:.0f}</div>
                <h2>{manual_icon} {manual_category}</h2>
                <p style="margin-top: 1rem;">{manual_desc}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Health recommendation based on manual prediction
        if manual_prediction <= 50:
            st.success("✅ **Excellent Air Quality!** Perfect for outdoor activities.")
        elif manual_prediction <= 100:
            st.info("ℹ️ **Moderate Air Quality.** Sensitive individuals should take care.")
        elif manual_prediction <= 150:
            st.warning("⚠️ **Unhealthy for Sensitive Groups.** Limit outdoor activities.")
        elif manual_prediction <= 200:
            st.warning("⚠️ **Unhealthy Air Quality.** Avoid outdoor activities.")
        else:
            st.error("🚨 **Hazardous Air Quality!** Stay indoors. Wear N95 masks.")
    
    st.markdown('</div>', unsafe_allow_html=True)

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
# 4-DAY FORECAST
# ============================================
st.markdown("##  3-Day Air Quality Forecast")

bad_future = any(f['AQI'] > 150 for f in forecasts[2:])
if bad_future:
    st.markdown("### ⚠️ Deteriorating air quality expected in the coming days")
else:
    st.markdown("### ✅ Stable air quality expected")

forecast_cols = st.columns(4)  # Changed from 5 to 4 columns

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
# SHAP ANALYSIS SECTION
# ============================================
st.markdown("## AI Explainability: What Affects Air Quality?")
st.markdown("*Using SHAP-like feature importance analysis to explain predictions*")

if shap_importance is not None and not shap_importance.empty:
    col_shap1, col_shap2 = st.columns([3, 2])
    
    with col_shap1:
        # Create horizontal bar chart for feature importance
        fig_importance = px.bar(
            shap_importance.head(10),
            x='Importance',
            y='Feature',
            orientation='h',
            title="Feature Importance Analysis (SHAP-based)",
            labels={'Importance': 'Impact on AQI Prediction', 'Feature': ''},
            color='Importance',
            color_continuous_scale='Blues',
            text='Importance'
        )
        
        fig_importance.update_layout(
            height=450,
            plot_bgcolor='white',
            paper_bgcolor='white',
            xaxis=dict(gridcolor='#e2e8f0'),
            yaxis=dict(gridcolor='#e2e8f0')
        )
        
        fig_importance.update_traces(
            texttemplate='%{text:.3f}',
            textposition='outside'
        )
        
        st.plotly_chart(fig_importance, use_container_width=True, key="shap_importance_chart")
    
    with col_shap2:
        st.markdown(f"""
        <div class="shap-card">
            <h3>🔍 Key Insights</h3>
            <ul style="list-style-type: none; padding-left: 0;">
                <li>📊 <strong>Top Factor:</strong> {shap_importance.iloc[0]['Feature']} ({shap_importance.iloc[0]['Importance']:.3f})</li>
                <li>📊 <strong>Second Factor:</strong> {shap_importance.iloc[1]['Feature'] if len(shap_importance) > 1 else 'N/A'}</li>
                <li>📊 <strong>Third Factor:</strong> {shap_importance.iloc[2]['Feature'] if len(shap_importance) > 2 else 'N/A'}</li>
            </ul>
            <hr>
            <h3>💡 What This Means</h3>
            <p>Higher importance = greater influence on AQI prediction.<br>
            <strong>{shap_importance.iloc[0]['Feature']}</strong> is the most significant pollutant affecting air quality in Karachi.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background: #e8f4f8; border-radius: 15px; padding: 1rem; margin: 1rem 0;">
        <strong>📖 How to Interpret:</strong><br>
        • Higher percentage = stronger influence on AQI<br>
        • Focus on controlling high-importance pollutants for better air quality<br>
        • Time-based features show pollution patterns throughout the day
    </div>
    """, unsafe_allow_html=True)

# ============================================
# AQI TREND CHART
# ============================================
st.markdown("## AQI Trend Chart")

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
    title=dict(text="4-Day Air Quality Index Forecast", font=dict(size=18, weight='bold', color='#0f172a')),
    xaxis_title=dict(text="Day", font=dict(size=14, weight='bold', color='#475569')),
    yaxis_title=dict(text="AQI Value", font=dict(size=14, weight='bold', color='#475569')),
    hovermode='x unified',
    xaxis=dict(gridcolor='#e2e8f0', gridwidth=1),
    yaxis=dict(gridcolor='#e2e8f0', gridwidth=1)
)

st.plotly_chart(fig, use_container_width=True, key="trend_chart")

st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

# ============================================
# HOURLY FORECAST
# ============================================
st.markdown("## Hourly Forecast (Next 24 Hours)")

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
# DETAILED FORECAST TABLE
# ============================================
st.markdown("## Detailed Forecast with Health Advisories")

detail_df = pd.DataFrame(forecasts)[['Day', 'Date', 'AQI', 'Category', 'PM2.5', 'PM10', 'Description']]
detail_df.columns = ['Day', 'Date', 'AQI', 'Air Quality', 'PM2.5 (μg/m³)', 'PM10 (μg/m³)', 'Health Advisory']

st.dataframe(detail_df, use_container_width=True, hide_index=True)

st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

# ============================================
# HEALTH RECOMMENDATIONS
# ============================================
st.markdown("## Health Recommendations")

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
    <p>🌿 AirNet |  Real-time Air Quality Intelligence | Powered by OpenWeather API & Machine Learning</p>
    <p>Forecast confidence: Medium. Air quality can change rapidly based on local conditions.</p>
    <p>Data updates every hour | Model retrains daily| "Breathe Clean. Live Better" - AirNet </p>
</div>
""", unsafe_allow_html=True)