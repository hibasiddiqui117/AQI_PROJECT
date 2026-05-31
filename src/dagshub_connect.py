import dagshub
import pandas as pd

# Connect to DagsHub repo
dagshub.init(repo_owner="hibasiddiqui117", repo_name="AQI_PROJECT", mlflow=True)

print(" Connected to DagsHub")

# Load dataset
df = pd.read_csv("data/processed_aqi_data.csv")

# Save dataset into DagsHub (acts like feature store)
df.to_csv("data/dagshub_features.csv", index=False)

print(" Data saved to DagsHub format")