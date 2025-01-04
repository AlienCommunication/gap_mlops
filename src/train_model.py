import os
import pickle
import pandas as pd
import numpy as np
from google.cloud import storage, aiplatform
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import config.settings as cfg

# Set up GCP clients
storage_client = storage.Client()
aiplatform.init(project=cfg.PROJECT_ID, location=cfg.REGION)

# Load processed data from Cloud Storage
bucket = storage_client.bucket(cfg.BUCKET_NAME)
blob = bucket.blob("processed_data/california_housing.csv")
df = pd.read_csv(blob.download_as_text())

# Define features & target
X = df.drop(columns=["median_house_value"])
y = df["median_house_value"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"Model Mean Absolute Error: {mae}")

# Save model locally
model_filename = "random_forest_model.pkl"
with open(model_filename, "wb") as f:
    pickle.dump(model, f)

# Upload model to Cloud Storage
model_blob = bucket.blob(f"models/{model_filename}")
model_blob.upload_from_filename(model_filename)
print(f"Model saved to: gs://{cfg.BUCKET_NAME}/models/{model_filename}")

# Upload model to Vertex AI Model Registry
model_artifact = aiplatform.Model.upload(
    display_name=cfg.MODEL_NAME,
    artifact_uri=f"gs://{cfg.BUCKET_NAME}/models/{model_filename}",
    serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-0:latest",
)

print(f"Model uploaded to Vertex AI: {model_artifact.resource_name}")
