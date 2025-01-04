import os
import pickle
import pandas as pd
import numpy as np
import os
import sys
import io
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from google.cloud import storage, aiplatform
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from config import settings as cfg
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.abspath("gcp-key.json")

# Set up GCP clients
storage_client = storage.Client()
aiplatform.init(project=cfg.PROJECT_ID, location=cfg.REGION)

# Load processed data from Cloud Storage
bucket = storage_client.bucket(cfg.BUCKET_NAME.replace("gs://", ""))
blob = bucket.blob("processed_data/nyc_taxi_data.csv")

try:
    csv_data = blob.download_as_text()
    df = pd.read_csv(io.StringIO(csv_data))
    print("✅ Successfully loaded preprocessed NYC taxi data from Cloud Storage.")
except Exception as e:
    print("❌ Failed to load file from Cloud Storage. Ensure preprocessing runs first.")
    raise e

# Convert datetime columns to proper format
df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
df["dropoff_datetime"] = pd.to_datetime(df["dropoff_datetime"])

# Compute trip duration in minutes (New Feature)
df["trip_duration"] = (df["dropoff_datetime"] - df["pickup_datetime"]).dt.total_seconds() / 60

# Define features & target
X = df.drop(columns=["fare_amount", "pickup_datetime", "dropoff_datetime"])  # ✅ Remove datetime columns
y = df["fare_amount"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"✅ Model Mean Absolute Error: {mae}")

# Save model locally
model_filename = "random_forest_nyc_taxi.pkl"
with open(model_filename, "wb") as f:
    pickle.dump(model, f)

# Upload model to Cloud Storage
model_blob = bucket.blob(f"models/{model_filename}")
model_blob.upload_from_filename(model_filename)
print(f"✅ Model saved to: gs://{cfg.BUCKET_NAME}/models/{model_filename}")

# Upload model to Vertex AI Model Registry
model_artifact = aiplatform.Model.upload(
    display_name=cfg.MODEL_NAME,
    artifact_uri=f"gs://{cfg.BUCKET_NAME}/models/{model_filename}",
    serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-0:latest",
)

print(f"✅ Model uploaded to Vertex AI: {model_artifact.resource_name}")