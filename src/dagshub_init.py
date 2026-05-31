import dagshub

# Connect your repo to DagsHub
dagshub.init(
    repo_owner="hibasiddiqui117",
    repo_name="AQI_PROJECT",
    mlflow=True
)
print(" Dagshub connected successfully")