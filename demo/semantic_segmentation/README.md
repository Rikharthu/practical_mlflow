# Semantic Segmentation

[Semantic Segmentation](https://lightning-flash.readthedocs.io/en/latest/reference/semantic_segmentation.html)

Create environment manually:

```shell
conda env create -n semantic_segmentation --file ./conda.yaml
```

Prepare environnment
```shell
export MLFLOW_TRACKING_URI=http://192.168.0.97
export MLFLOW_S3_ENDPOINT_URL=http://192.168.0.97:9000
export AWS_ACCESS_KEY_ID="minio"
export AWS_SECRET_ACCESS_KEY="minio123"
```

Locally:

```shell
export MLFLOW_TRACKING_URI=http://localhost
export MLFLOW_S3_ENDPOINT_URL=http://localhost:9000
export AWS_ACCESS_KEY_ID="minio"
export AWS_SECRET_ACCESS_KEY="minio123"
```

### Running as MLproject

We reorganized code to be machine learning pipeline that can be run with MLflow as an MLproject.

```shell
mlflow run . --experiment-name semantic_segmentation_project -P pipeline_steps=all
```

```shell
mlflow run . --experiment-name semantic_segmentation_project -P pipeline_steps=train_model,download_model
```