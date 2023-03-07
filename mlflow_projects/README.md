# MLflow Projects

Copied from https://github.com/aruberts/tutorials/tree/main/mlflow_project

Configure enviornment:

```shell
export MLFLOW_TRACKING_URI=http://192.168.0.97
export MLFLOW_S3_ENDPOINT_URL=http://192.168.0.97:9000
export AWS_ACCESS_KEY_ID="minio"
export AWS_SECRET_ACCESS_KEY="minio123"
```

Run experiment:

```shell
bash run.sh
```