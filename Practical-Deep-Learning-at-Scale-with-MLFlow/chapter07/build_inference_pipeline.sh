MLFLOW_TRACKING_URI=http://localhost \
  MLFLOW_S3_ENDPOINT_URL=http://localhost:9000 \
  AWS_ACCESS_KEY_ID="minio" \
  AWS_SECRET_ACCESS_KEY="minio123" \
  mlflow run . \
  -e inference_pipeline_model \
  --experiment-name dl_model_chapter07 \
  -P finetuned_model_run_id=d31abc1287964a6599ce8a6ddc392f4e
