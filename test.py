import requests
import pandas as pd
import os
from datetime import datetime

# Test 1: Check if we can import packages
print("✅ Test 1: Package imports successful")

# Test 2: Check API key
API_KEY = input("Enter your OpenWeather API key: ")
# Or use environment variable
# API_KEY = os.getenv("OPENWEATHER_API_KEY")

LAT = "24.8607"
LON = "67.0011"

url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={LAT}&lon={LON}&appid={API_KEY}"

print(f"\n🌐 Testing API connection...")

try:
    response = requests.get(url, timeout=10)
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        print("✅ API Test PASSED!")
        data = response.json()
        print(f"Current AQI: {data['list'][0]['main']['aqi']}")
        print(f"PM2.5: {data['list'][0]['components']['pm2_5']}")
    else:
        print(f"❌ API Test FAILED: {response.text}")
        
except Exception as e:
    print(f"❌ Error: {e}")

# Test 3: Check if we can write to data directory
print("\n📁 Testing file operations...")
os.makedirs("data", exist_ok=True)

test_df = pd.DataFrame([{"test": 1, "datetime": datetime.now()}])
test_df.to_csv("data/test.csv", index=False)

if os.path.exists("data/test.csv"):
    print("✅ File write test PASSED!")
    os.remove("data/test.csv")
else:
    print("❌ File write test FAILED!")