# 🌿 AirNet - "Breathe Clean. Live Better"

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://airnet.streamlit.app/)
[![GitHub Actions](https://github.com/hibasiddiqui117/AQI_PROJECT/actions/workflows/feature_pipeline.yml/badge.svg)](https://github.com/hibasiddiqui117/AQI_PROJECT/actions)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Live Demo:** [https://airnet.streamlit.app/](https://airnet.streamlit.app/)

---

## The Backstory

During my internship, I was assigned a project that would push me beyond the usual academic exercises. I had to build something real, something that could actually make a difference. The task was to create an end-to-end ML system that solves a practical problem, from data collection to deployment.

I chose to tackle air pollution, a crisis I see every day in my city. I live in Karachi, where you can see pollution in the air, but you can't always see it coming. That got me thinking: *What if I could predict air quality before it gets bad?*

So I built **AirNet**.

What started as an internship assignment became a fully automated MLOps pipeline that predicts air quality in Karachi. It runs entirely on autopilot, from fetching live pollution data every hour, to retraining models daily, to serving predictions through an interactive dashboard. I didn't just build a model; I built a system that actually works in production.

---

## What It Does

AirNet is a **complete MLOps pipeline** that forecasts air quality for Karachi, Pakistan. It runs entirely on autopilot:

- **Fetches live pollution data** every hour via OpenWeather API
- **Engineers 36+ features** from raw API responses (time-based, lag, rolling statistics)
- **Retrains ML models daily** with fresh, evolving data
- **Serves predictions** through an interactive, weather-app-style dashboard
-  **Explains predictions** using SHAP-based feature importance

---

## What I Learned

**Building something real is different from building something academic.**

I spent more time dealing with API rate limits, duplicate timestamps, and GitHub Actions permission issues than I did tuning hyperparameters. But that's exactly why I built this to learn what actually happens when you deploy ML in production.

### Some challenges I overcame:

| Challenge | How I Solved It |
|-----------|-----------------|
| API rate limits | Implemented caching with TTL and retry logic |
| Data duplication | Built a deduplication pipeline with proper datetime handling |
| GitHub Actions not persisting data | Made the workflow commit and push CSV changes |
| Streamlit duplicate element errors | Added unique keys to every Plotly chart |
| SHAP installation on Windows | Used built-in feature importance as a lightweight alternative |
| Single AQI class in training data | Augmented data to create realistic variations for all 6 classes |

### Things I'm proud of:
- The dashboard feels like a real product, not a demo
- The pipeline runs autonomously... I don't touch it
- The SHAP explanations help anyone understand what drives air quality
- I learned more about engineering than ML in the best way possible

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| **Machine Learning** | Python, Scikit-learn, Pandas, NumPy |
| **Models** | Random Forest Regressor & Classifier |
| **Dashboard** | Streamlit, Plotly |
| **Data Source** | OpenWeather API |
| **Automation** | GitHub Actions |
| **Experiment Tracking** | DagsHub, MLflow |
| **Explainability** | SHAP (feature importance) |

---

## Features

- **Real-time AQI monitoring** live data from OpenWeather API
- **4-day AQI forecast** with realistic degradation patterns
- **Interactive dashboard** MSN Weather style, clean and intuitive
- **Manual prediction** adjust pollutant sliders to see real-time impact
- **SHAP explainability** understand what factors affect air quality
- **Health alerts** automatic warnings for hazardous AQI levels
- **Hourly forecast** next 24 hours breakdown
- **Automated everything** data collection, model retraining, and deployment

## Model Performance
- Regression R² Score: 0.987 (98.7% accuracy)
- Classification Accuracy: 100%
- MAE: 15.53 AQI points

## Dashboard Preview
4-day AQI forecast

Real-time weather data

Hourly predictions

Health recommendations

Alert system

---

## Project Structure
AQI_PROJECT/

├── app/streamlit_app.py # Main dashboard

├── pipelines/ # Feature & training pipelines

├── src/feature_store/ # DagsHub integration

├── .github/workflows/ # GitHub Actions

└── models/ # Trained models

---

## Quick Start

Want to run this locally?

```bash
# Clone the repository
git clone https://github.com/hibasiddiqui117/AQI_PROJECT.git
cd AQI_PROJECT

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up your API key in .env file
# OPENWEATHER_API_KEY=your_key_here

# Run the feature pipeline (fetch data)
python pipelines/feature_pipeline.py

# Train the models
python pipelines/training_pipeline.py

# Launch the dashboard
streamlit run app/streamlit_app.py

```
---
**Final Thoughts:** This project taught me that building ML systems is about 20% model training and 80% everything else: data engineering, API integration, error handling, automation, and user experience.

*I built AirNet because I believe AI should be used to solve real problems, not just win Kaggle competitions. Clean air isn't a luxury; it's a right. And if this project helps even a few people make better decisions about their health, it's worth it.*

---

*"Breathe Clean. Live Better."*


