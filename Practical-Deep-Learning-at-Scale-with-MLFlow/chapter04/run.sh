MLFLOW_TRACKING_URI=http://192.168.0.96 \
  MLFLOW_S3_ENDPOINT_URL=http://192.168.0.96:9000 \
  AWS_ACCESS_KEY_ID="minio" \
  AWS_SECRET_ACCESS_KEY="minio123" \
  PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
  mlflow run . \
  --experiment-name=dl_model_chapter04 \
  -P pipeline_steps=download_data,fine_tuning_model,register_model
