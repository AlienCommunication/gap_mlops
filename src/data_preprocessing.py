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

query = """
    SELECT 
        pickup_datetime, dropoff_datetime, trip_distance, fare_amount, passenger_count 
    FROM `bigquery-public-data.new_york_taxi_trips.tlc_yellow_trips_2022`
    WHERE trip_distance > 0 AND fare_amount > 0
    LIMIT 5000
"""

# Load data
df = client.query(query).to_dataframe()

# Handle missing values
df["total_bedrooms"].fillna(df["total_bedrooms"].median(), inplace=True)

# Save processed data to Cloud Storage
bucket = storage.Client().bucket(cfg.BUCKET_NAME.replace("gs://", ""))
blob = bucket.blob("processed_data/california_housing.csv")
blob.upload_from_string(df.to_csv(index=False), "text/csv")

print("Preprocessed data uploaded to Cloud Storage")
