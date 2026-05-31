import pandas as pd

df = pd.read_csv("data/processed_aqi_data.csv")

print(df["aqi"].value_counts())