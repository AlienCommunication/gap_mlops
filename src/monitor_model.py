from google.cloud import aiplatform
import config.settings as cfg

# Initialize Vertex AI
aiplatform.init(project=cfg.PROJECT_ID, location=cfg.REGION)

# Get the latest model endpoint
endpoints = aiplatform.Endpoint.list(filter=f"display_name={cfg.MODEL_NAME}-endpoint", order_by="update_time desc")
if not endpoints:
    raise ValueError("No deployed endpoints found for monitoring.")

endpoint = endpoints[0]

# Enable Model Monitoring
job = aiplatform.ModelMonitoringJob.create(
    display_name=f"{cfg.MODEL_NAME}-monitoring-job",
    endpoint_name=endpoint.resource_name,
    analysis_instance_schema_uri="gs://google-cloud-aiplatform/schema/dataset/schema_v0.1.yaml",
    objective_configs=[
        {
            "feature_drift_thresholds": {
                "median_income": 0.05,  # Adjust based on acceptable drift
                "total_rooms": 0.05
            }
        }
    ],
    logging_sampling_strategy={"random_sample_config": {"sample_rate": 0.2}},  # Sample 20% of predictions
)

print(f"Model monitoring enabled: {job.resource_name}")
