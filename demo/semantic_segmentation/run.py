# TODO: make it a pipeline step that will log sample images as artifacts
import mlflow
import os

MLFLOW_RUN_ID = "cdffa48834734e079b303f85ab4cec99"

# Load PyTorch model
MODEL_ARTIFACT_PATH = "segmentation_model"

model_uri = f"runs:/{MLFLOW_RUN_ID}/{MODEL_ARTIFACT_PATH}"

model = mlflow.pytorch.load_model(model_uri)
print(f"Loaded model type: {type(model)}")
# -> Loaded model type: <class 'flash.image.segmentation.model.SemanticSegmentation'>

# Load ONNX model artifacts as files
MODEL_ONNX_ARTIFACT_PATH = "segmentation_model_onnx"
MODEL_ONNX_ARTIFACT_LOCAL_PATH = "./"

model_onnx_uri = f"runs:/{MLFLOW_RUN_ID}/{MODEL_ONNX_ARTIFACT_PATH}"

loaded_artifacts_directory = mlflow.artifacts.download_artifacts(
    model_onnx_uri,
    dst_path=MODEL_ONNX_ARTIFACT_LOCAL_PATH
)

model_onnx_local_path = os.path.join(MODEL_ONNX_ARTIFACT_LOCAL_PATH, MODEL_ONNX_ARTIFACT_PATH, "model.onnx")
print(f"Downloaded {model_onnx_local_path} exists: {os.path.exists(model_onnx_local_path)}")
# -> Downloaded ./segmentation_model_onnx/model.onnx exists: True
