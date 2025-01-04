import time
from google.cloud import aiplatform
import os
import sys
import io
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import settings as cfg
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.abspath("gcp-key.json")

# Initialize Vertex AI
aiplatform.init(project=cfg.PROJECT_ID, location=cfg.REGION)

# Load latest model from Vertex AI Model Registry
models = aiplatform.Model.list(filter=f"display_name={cfg.MODEL_NAME}", order_by="update_time desc")
if not models:
    raise ValueError("No models found in Vertex AI Model Registry.")

latest_model = models[0]

# Deploy model to Vertex AI Endpoints
endpoint = aiplatform.Endpoint.create(display_name=f"{cfg.MODEL_NAME}-endpoint")

deployed_model = latest_model.deploy(
    endpoint=endpoint,
    machine_type="n1-standard-4",  # Adjust machine size if needed
    traffic_split={"0": 100},
)

print(f"Model deployed to Endpoint: {endpoint.resource_name}")
