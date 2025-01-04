from google.cloud import bigquery
import pandas as pd
from google.cloud import storage
import config.settings as cfg

# Initialize BigQuery client
client = bigquery.Client()

# SQL Query to load data
query = f"""
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