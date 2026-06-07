# 🌿 AirNet - "Breathe Clean. Live Better"
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://airnet.streamlit.app/)

## Live Demo
 **Try the live application:** [AirNet AQI Dashboard]([https://your-app-name.streamlit.app](https://airnet.streamlit.app/))
## Overview
AirNet is an end-to-end MLOps pipeline for real-time Air Quality Index (AQI) forecasting for Karachi, Pakistan.

## Features
- Real-time AQI data from OpenWeather API
- 3-day AQI forecast with degradation patterns
- Machine learning models (Random Forest with 98.7% accuracy)
- Interactive MSN Weather-style dashboard
- Automated data collection (every hour)
- Daily model retraining
- DagsHub feature store integration
- SHAP explainability support

## Model Performance
- Regression R² Score: 0.987 (98.7% accuracy)
- Classification Accuracy: 100%
- MAE: 15.53 AQI points

## Tech Stack
- Python, Scikit-learn, Pandas, NumPy
- Streamlit for dashboard
- OpenWeather API
- DagsHub + MLflow for tracking
- GitHub Actions for automation

## Dashboard Preview
5-day AQI forecast

Real-time weather data

Hourly predictions

Health recommendations

Alert system

## Project Structure
AQI_PROJECT/

├── app/streamlit_app.py # Main dashboard

├── pipelines/ # Feature & training pipelines

├── src/feature_store/ # DagsHub integration

├── .github/workflows/ # GitHub Actions

└── models/ # Trained models

## Quick Start
```bash
# Install dependencies
pip install -r requirements.txt

# Run feature pipeline
python pipelines/feature_pipeline.py

# Train models
python pipelines/training_pipeline.py

# Launch dashboard
streamlit run app/streamlit_app.py

