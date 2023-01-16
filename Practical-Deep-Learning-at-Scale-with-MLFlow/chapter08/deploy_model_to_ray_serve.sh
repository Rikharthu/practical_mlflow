mlflow deployments create \
  -t ray-serve \
  -m runs:/97bd8d499a8c4bbfaab5433002c3e679/inference_pipeline_model \
  --name dl-inference-model-on-ray \
  -C num_replicas=1