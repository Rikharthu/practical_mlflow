MLFLOW_TRACKING_URI=http://localhost \
  MLFLOW_S3_ENDPOINT_URL=http://localhost:9000 \
  AWS_ACCESS_KEY_ID="minio" \
  AWS_SECRET_ACCESS_KEY="minio123" \
  python notebooks/multistep_inference_model.py
