from google.cloud import bigquery
import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from google.cloud import storage
from config import settings as cfg
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.abspath("gcp-key.json")
# Initialize BigQuery client
client = bigquery.Client()

# Query for NYC taxi dataset
query = """
    SELECT 
        pickup_datetime, dropoff_datetime, trip_distance, fare_amount, passenger_count 
    FROM `bigquery-public-data.new_york_taxi_trips.tlc_yellow_trips_2022`
    WHERE trip_distance > 0 AND fare_amount > 0
    LIMIT 5000
"""

# Run query and load data into Pandas DataFrame
df = client.query(query).to_dataframe()

# Print column names for debugging
print("✅ Columns in dataset:", df.columns)

# Handle missing values
df["passenger_count"].fillna(1, inplace=True)  # Replace missing passenger count with 1

# Convert datetime columns to proper format
df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
df["dropoff_datetime"] = pd.to_datetime(df["dropoff_datetime"])

# Compute trip duration in minutes
df["trip_duration"] = (df["dropoff_datetime"] - df["pickup_datetime"]).dt.total_seconds() / 60

# Save processed data to Cloud Storage
bucket = storage.Client().bucket(cfg.BUCKET_NAME.replace("gs://", ""))
blob = bucket.blob("processed_data/nyc_taxi_data.csv")
blob.upload_from_string(df.to_csv(index=False), "text/csv")

print("✅ Preprocessed NYC taxi data uploaded to Cloud Storage!")
