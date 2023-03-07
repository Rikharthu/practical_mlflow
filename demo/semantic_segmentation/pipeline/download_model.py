import mlflow
import os
import click

MODEL_ARTIFACT_PATH = "segmentation_model"

@click.command()
@click.option("--run_id", type=str)
def download(run_id):

    with mlflow.start_run(run_name="download_model", nested=True):
        model_uri = f"runs:/{run_id}/{MODEL_ARTIFACT_PATH}"
        print(f"Model URI: {model_uri}")

        model = mlflow.pytorch.load_model(model_uri)
        print(f"Loaded model type: {type(model)}")
        # -> Loaded model type: <class 'flash.image.segmentation.model.SemanticSegmentation'>

        # Load ONNX model artifacts as files
        MODEL_ONNX_ARTIFACT_PATH = "segmentation_model_onnx"
        MODEL_ONNX_ARTIFACT_LOCAL_PATH = "./"

        model_onnx_uri = f"runs:/{run_id}/{MODEL_ONNX_ARTIFACT_PATH}"

        loaded_artifacts_directory = mlflow.artifacts.download_artifacts(
            model_onnx_uri,
            dst_path=MODEL_ONNX_ARTIFACT_LOCAL_PATH
        )

        model_onnx_local_path = os.path.join(MODEL_ONNX_ARTIFACT_LOCAL_PATH, MODEL_ONNX_ARTIFACT_PATH, "model.onnx")
        print(f"Downloaded {model_onnx_local_path} exists: {os.path.exists(model_onnx_local_path)}")
        # -> Downloaded ./segmentation_model_onnx/model.onnx exists: True

        mlflow.log_param("downloaded_onnx_model_path", model_onnx_local_path)

if __name__ == "__main__":
    download()