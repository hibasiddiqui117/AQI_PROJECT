from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load trained regression model
model = joblib.load("models/regressor.pkl")

@app.route("/")
def home():
    return "AQI Prediction API Running Successfully"

# ------------------------------------------------
# PREDICTION ENDPOINT
# ------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():

    try:
        data = request.get_json()

        features = np.array([[
            data["co"],
            data["no"],
            data["no2"],
            data["o3"],
            data["so2"],
            data["pm2_5"],
            data["pm10"],
            data["nh3"],
            data["hour"],
            data["day"],
            data["month"],
            data["aqi_lag_1"],
            data["aqi_rolling_mean"],
            data["aqi_change"]
        ]])

        prediction = model.predict(features)[0]

        return jsonify({
            "predicted_pm2_5": float(prediction)
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        })

if __name__ == "__main__":
    app.run(debug=True)