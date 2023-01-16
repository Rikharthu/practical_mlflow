MLFLOW_TRACKING_URI=http://localhost \
  MLFLOW_S3_ENDPOINT_URL=http://localhost:9000 \
  AWS_ACCESS_KEY_ID="minio" \
  AWS_SECRET_ACCESS_KEY="minio123" \
  mlflow run . \
  --experiment-name=dl_model_chapter07 \
  -P pipeline_steps=download_data,fine_tuning_model
