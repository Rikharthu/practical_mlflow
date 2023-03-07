# Demo

### What is MLflow?

https://mlflow.org

* Open-source
* Part of **MLOps**, plays role of **ModelOps**
    * Machine learning lifecycle
* Tracking and managing experiments
    * Provenance
        * Seek which experiment launched a run
        * With which parameters was run started and which metrics & artifacts it produced
        * Versioning
        * Git commit
    * Compare
    * Log
* Models
    * http://192.168.0.97/#/models
* APIs
    * [Python](https://mlflow.org/docs/latest/python_api/index.html)
    * [R](https://mlflow.org/docs/latest/R-api.html)
    * [Java](https://mlflow.org/docs/latest/java_api/index.html)
    * [REST](https://mlflow.org/docs/latest/rest-api.html)

### Deployment

* Locally
    * `pip install mlflow`
    * `mlflow ui`
    * Stores artifacts and metadta locally in an `./mlruns` directory
* Production
    * Can configure Metadata database
        * PostgreSQL
        * MySQL
        * MSSQL
        * SQLite
    * Artifact store
        * Any S3-compatible object store
    * Demo: deployment using docker compose
        * [docker-compose.yml](../mlflow_docker_setup/docker-compose.yml)
        * `ssh richard@192.168.0.97 "docker ps"`
        * http://192.168.0.97

## Experiment Tracking

### Demo: manual logging in PyTorch

[torch_logging](./torch_logging)

```shell
conda create -n mlflow_torch_logging python=3.10 -y
conda activate mlflow_torch_logging
pip install -r requirements.txt
```

Configure MLflow endpoint (can also be done in code)

```shell
export MLFLOW_TRACKING_URI=http://192.168.0.97
export MLFLOW_S3_ENDPOINT_URL=http://192.168.0.97:9000
export AWS_ACCESS_KEY_ID="minio"
export AWS_SECRET_ACCESS_KEY="minio123"
```

Fine-tune whole model:

```shell
python train_ft.py
```

Open MLflow UI in the browser and explore the newly created **Run** and real-time metric updates

Use pre-trained model as feature extractor:

```shell
python train_conv.py 
```

Now we can compare this experiment runs and register the best model.

This model can then be downloaded through MLflow APIs or directly from S3.
We can also see which run produced the model, what were the training parameters, metrics, etc, so that experiment can be
easily replicated.

### Autologging

[Demo: semantic segmentation with PyTorch Lightning](./semantic_segmentation)

It doesn't work on my Mac, so will train on different machine:

```shell
ssh richard@192.168.0.97

cd ~/practical_mlflow/demo/semantic_segmentation

# Create conda environment
conda env create -n semantic_segmentation --file ./conda.yaml
conda activate semantic_segmentation

# It is the same machine as MLflow runs on, so configure mlflow client to connect to localhost:
export MLFLOW_TRACKING_URI=http://localhost
export MLFLOW_S3_ENDPOINT_URL=http://localhost:9000
export AWS_ACCESS_KEY_ID="minio"
export AWS_SECRET_ACCESS_KEY="minio123"

# Start training
python train.py

# Use trained model
python run.py
```

#### Running as MLproject

We reorganized code to be machine learning pipeline that can be run with MLflow as an MLproject.

Same training process, but with MLproject. Automatic environment creation (conda, virtualenv, etc).

```shell
mlflow run . --experiment-name semantic_segmentation_project -P pipeline_steps=all
```

```shell
mlflow run . --experiment-name semantic_segmentation_project -P pipeline_steps=train_model,download_model
```

Passed parameters will also be automatically logged by MLflow

Download trained ONNX model artifact from MLflow UI or as local artifact from `download_model` step and open it in
Netron

### Running MLprojects from Git

[Demo](./running_from_git/README.md)

## Hyperaparameter Tuning

TODO: link to that sample code from "Practical MLflow" book.

[chapter06](../Practical-Deep-Learning-at-Scale-with-MLFlow/chapter06)

Run on Linux ThinkCentre:

```shell
export MLFLOW_TRACKING_URI=http://localhost
export MLFLOW_S3_ENDPOINT_URL=http://localhost:9000
export AWS_ACCESS_KEY_ID="minio"
export AWS_SECRET_ACCESS_KEY="minio123"

# From chapter06 directory:
conda env create -n hpo-demo -f ./conda.yaml

conda activate hpo-demo

python pipeline/hpo_finetuning_model.py

python pipeline/hpo_finetuning_model_optuna.py
```

## Integrations & Other Capabilities

* Hyperparameter tuning: OpTuna and others
* Serving
    * Ray
    * BentoML
    * Others..
    * MLflow
      itself: [Serving an MLflow Model form Model Registry](https://mlflow.org/docs/latest/registry.html?highlight=serving#serving-an-mlflow-model-from-model-registry)
* Sample MLops workflows
    * [Using Airflow with Tensorflow and MLflow](https://youtu.be/4tRTfwqcuWU?t=691)
        * K8s
    * [Airflow + MLflow stack](https://mymlops.com/examples/airflow-mlflow)
        * Git with DVC
        * Airflow
        * MLflow
        * BentoML
        * Prometheus & Grafana
* Integrates with most machine learning & serving frameworks
    * Custom models can be easily integrated as well by implementing a couple of lifecycle methods (similarly to PyTorch
      Lightning)
        *
      Example: [multistep_inference_model.py](../Practical-Deep-Learning-at-Scale-with-MLFlow/chapter07/notebooks/multistep_inference_model.py)
* Quite popular
    * [MLflow and Azure Machine Learning](https://learn.microsoft.com/en-us/azure/machine-learning/concept-mlflow)

## TMP Use-cases

* Model registry, version and environment management
    * Download & update models
        * Sample naive code:  [rust_mlflow_rest_api_demo](../rust_mlflow_rest_api_demo)
            * Download latest up-to-date model from a specified environment
* Track & compare experiments, perform HPO
    * There are tutorials on integrating MLflow into Ultralytics YOLO models
        * https://medium.com/codex/setting-up-mlflow-for-ultralytics-yolov5-1380b5f8cac5
        * https://techblog.sms-digital.com/integrating-mlflow-into-ultralytics/yolov5-1
        * https://github.com/ultralytics/ultralytics/issues/199
* Build & launch machine learning pipelines
    * Automatically builds Docker container from training code, pushes to registry and launches Kubernetes Job.

## Notes

* It is very easy to use and can even be used for local development
    * Launch with 'mlflow ui'
        * It will automatically look in 'mlruns' by default.
        * Can configure 'MLFLOW_TRACKING_URI=http://localhost' to run from any directory
* Supports most popular ML frameworks
    * Also supports autologging
        * It doesn't work with raw PyTorch, as we have to write training loops manually there, but it works with
          Lightning and Lightning-Flash
            * And any PyTorch model can be wrapped in Lightning
    * Log models, artifacts, parameters, tags, metrics with simple calls
        * Manage model metadata
            * Signatures
                * Input and output name, shape, type, etc.
                    * Automatically used by MLflow serving
    * Unsupported frameworks and non-serializable models can be easily integrated with wrappers
        * Hooks for initialized context
* Pipelines & MLproject
* PyFunc model representation
    * Can even add extra functionality
* Automatically captures environment data
    * Conda, packages & versions installed.
    * Can also create & manage environments
* Provenance
* Model management
    * Staging, Production
* REST API
* Kubernetes Job
    * Not researched in-depth yet
    * Automatically builds Docker container from training code, pushes to registry and launches Kubernetes Job.
* Serving
* Integrates with hyperparameter tuning frameworks
    * Ray
    * OpTuna
* Quite popular
    * [MLflow and Azure Machine Learning](https://learn.microsoft.com/en-us/azure/machine-learning/concept-mlflow)

## Resources

* How to use with DVC
    * https://www.youtube.com/watch?v=W2DvpCYw22o
* [Hyper-parameter optimization using HyperOpt and nested runs in MLflow](https://github.com/Azure/azureml-examples/blob/main/sdk/python/using-mlflow/train-and-log/xgboost_nested_runs.ipynb)
* [Logging models with MLflow](https://github.com/Azure/azureml-examples/blob/main/sdk/python/using-mlflow/train-and-log/logging_and_customizing_models.ipynb)

## Tracking

* Tracks and log experiments
* Profenance
* Manual and autologging for some frameworks
* Log file artifacts
* Automatically logs execution errors

## MLFlow Projects

* MLFlow project definition file
    * [Environments](https://mlflow.org/docs/latest/projects.html#project-environments) & Dependencies
        * System
        * Docker
        * Conda
        * Virtualenv
    * Pipelines
* Can be run from Git
    * `mlflow run https://github.com/mlflow/mlflow-example.git -P alpha=5.0`
* Can be run in Kubernetes
    * [Run an MLflow Project on Kubernetes](https://mlflow.org/docs/latest/projects.html#run-an-mlflow-project-on-kubernetes)
    * Supports job templates for reusability
* Pipeline: can contain multiple steps
    * [Building Multistep Workflows](https://mlflow.org/docs/latest/projects.html#building-multistep-workflows)
        * [Example](https://github.com/mlflow/mlflow/tree/master/examples/multistep_workflow)
    * Reusability of workflows
* Articles describing how MLflow can be used with ultralytics YOLO models

## Custom Models

[Example: Saving an XGBoost model in MLflow format
](https://mlflow.org/docs/latest/models.html?highlight=load_context#example-saving-an-xgboost-model-in-mlflow-format)

* Wraps a custom non-serializable model in a MLflow PyFuncModel, saves its data as artifact and loads it form MLflow
  in `load_context()` when needed to reconstruct the model.

[chapter07/notebooks/multistep_inference_model.py](../Practical-Deep-Learning-at-Scale-with-MLFlow/chapter07/notebooks/multistep_inference_model.py)

* Loads custom model parameters from MLflow (non-serializable model URI) and uses it to initialize custom model
  in `load_context()`