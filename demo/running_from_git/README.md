# Running MLflow Projects from Git

Based on this quickstart
section: [Running MLflow Projects](https://mlflow.org/docs/latest/quickstart.html#running-mlflow-projects)

Now we will run a modified version of example MLflow project from
GitHub - [mlflow-example](https://github.com/Rikharthu/mlflow-example).
It will create a conda environment and pass the parameters we specify.

```shell
export MLFLOW_TRACKING_URI=http://192.168.0.97
export MLFLOW_S3_ENDPOINT_URL=http://192.168.0.97:9000
export AWS_ACCESS_KEY_ID="minio"
export AWS_SECRET_ACCESS_KEY="minio123"
mlflow run --experiment-name Git_ElasticNet https://github.com/Rikharthu/mlflow-example -P alpha=5.0
```

We can then run the trained MLflow model through the API, use the artifacts, compare experiments etc.
We can also register the model to be used by our systems.

We can also re-run the experiments with better parameters:

```shell
mlflow run --experiment-name Git_ElasticNet https://github.com/Rikharthu/mlflow-example -P alpha=7.0 -P l1_ratio=0.15
```

We can also compare multiple different hyperparameters and their effects in runs. Here we use shell to launch multiple
runs, but we can do so in code and even integrate
hyperparameter tuning framework.

```shell
mlflow run --experiment-name Git_ElasticNet https://github.com/Rikharthu/mlflow-example -P alpha=20.0 -P l1_ratio=0.15
mlflow run --experiment-name Git_ElasticNet https://github.com/Rikharthu/mlflow-example -P alpha=15.0 -P l1_ratio=0.15
mlflow run --experiment-name Git_ElasticNet https://github.com/Rikharthu/mlflow-example -P alpha=10.0 -P l1_ratio=0.15
mlflow run --experiment-name Git_ElasticNet https://github.com/Rikharthu/mlflow-example -P alpha=7.0 -P l1_ratio=0.15
mlflow run --experiment-name Git_ElasticNet https://github.com/Rikharthu/mlflow-example -P alpha=4.0 -P l1_ratio=0.15
mlflow run --experiment-name Git_ElasticNet https://github.com/Rikharthu/mlflow-example -P alpha=3.0 -P l1_ratio=0.15
mlflow run --experiment-name Git_ElasticNet https://github.com/Rikharthu/mlflow-example -P alpha=2.0 -P l1_ratio=0.15
mlflow run --experiment-name Git_ElasticNet https://github.com/Rikharthu/mlflow-example -P alpha=20.0 -P l1_ratio=0.7
mlflow run --experiment-name Git_ElasticNet https://github.com/Rikharthu/mlflow-example -P alpha=15.0 -P l1_ratio=0.7
mlflow run --experiment-name Git_ElasticNet https://github.com/Rikharthu/mlflow-example -P alpha=10.0 -P l1_ratio=0.7
mlflow run --experiment-name Git_ElasticNet https://github.com/Rikharthu/mlflow-example -P alpha=7.0 -P l1_ratio=0.7
mlflow run --experiment-name Git_ElasticNet https://github.com/Rikharthu/mlflow-example -P alpha=4.0 -P l1_ratio=0.7
mlflow run --experiment-name Git_ElasticNet https://github.com/Rikharthu/mlflow-example -P alpha=3.0 -P l1_ratio=0.7
mlflow run --experiment-name Git_ElasticNet https://github.com/Rikharthu/mlflow-example -P alpha=2.0 -P l1_ratio=0.7
```

When running on local machine:

```shell
export MLFLOW_TRACKING_URI=http://localhost
export MLFLOW_S3_ENDPOINT_URL=http://localhost:9000
export AWS_ACCESS_KEY_ID="minio"
export AWS_SECRET_ACCESS_KEY="minio123"
```