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

# SQL Query to load data
query = query = f"""
    SELECT longitude, latitude, housing_median_age, total_rooms, 
           total_bedrooms, population, households, median_income, 
           median_house_value 
    FROM `{cfg.BQ_DATASET}.{cfg.BQ_TABLE}`
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
