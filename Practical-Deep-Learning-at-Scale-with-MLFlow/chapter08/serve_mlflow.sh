export MLFLOW_TRACKING_URI=http://localhost
export MLFLOW_S3_ENDPOINT_URL=http://localhost:9000
export AWS_ACCESS_KEY_ID="minio"
export AWS_SECRET_ACCESS_KEY="minio123"

mlflow models serve --env-manager conda -m models:/inference_pipeline_model/1

# Request predictions:
# curl http://127.0.0.1:5000/invocations -H 'Content-Type: application/json' -d '{
#     "dataframe_split": {
#       "columns": ["text"],
#       "data": [["This is the best movie we saw."], ["What a movie!"]]
#     }
# }'

# Sample response:
# {"predictions": [{"text": "{\"response\": {\"prediction_label\": [\"negative\"]}, \"metadata\": {\"language_detected\": \"en\"}, \"model_metadata\": {\"finetuned_model_uri\": \"runs:/d31abc1287964a6599ce8a6ddc392f4e/model\", \"inference_pipeline_model_uri\": \"runs:/97bd8d499a8c4bbfaab5433002c3e679/inference_pipeline_model\"}}"}, {"text": "{\"response\": {\"prediction_label\": [\"negative\"]}, \"metadata\": {\"language_detected\": \"en\"}, \"model_metadata\": {\"finetuned_model_uri\": \"runs:/d31abc1287964a6599ce8a6ddc392f4e/model\", \"inference_pipeline_model_uri\": \"runs:/97bd8d499a8c4bbfaab5433002c3e679/inference_pipeline_model\"}}"}]}