import os
import mlflow

os.environ["MLFLOW_TRACKING_URI"] = "http://localhost"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"
os.environ["AWS_ACCESS_KEY_ID"] = "minio"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minio123"

# Get it from the mlflow model registry.
model_name = "dl_finetuned_model"
model_version = 1

print(f"Loading model {model_name} version {model_version} from the registry")
# Load model from the registry
model = mlflow.pyfunc.load_model(
    f"models:/{model_name}/{model_version}"
)
print(model)
print(type(model))
print(model.metadata)
print(model.metadata.run_id) # a3e102f5020a471daaed6c2dfc1d9c6a

# We can also download the same model using it's run id:
run_id = "a3e102f5020a471daaed6c2dfc1d9c6a"
logged_model = f"runs:/{run_id}/model"
print(f"Loading logged model {logged_model}")
# We can also load it as native PyTorch model isntead of PyFunc
# model2 = mlflow.pytorch.load_model(logged_model)
model2 = mlflow.pyfunc.load_model(logged_model)
print(model2)


# predictions = model.predict(data)