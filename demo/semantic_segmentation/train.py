# https://lightning-flash.readthedocs.io/en/stable/reference/semantic_segmentation.html

import torch

import flash
from flash.core.data.utils import download_data
from flash.image import SemanticSegmentation, SemanticSegmentationData
import mlflow
import onnx

EXPERIMENT_NAME = "semantic_segmentation"
mlflow.set_experiment \
    (EXPERIMENT_NAME)

mlflow.pytorch.autolog(
    # We want to log models explicitly
    log_models=False
)

# The data was generated with the  CARLA self-driving simulator as part of the Kaggle Lyft Udacity Challenge.
# More info here: https://www.kaggle.com/kumaresanmanickavelu/lyft-udacity-challenge
download_data(
    "https://github.com/ongchinkiat/LyftPerceptionChallenge/releases/download/v0.1/carla-capture-20180513A.zip",
    "./data",
)

datamodule = SemanticSegmentationData.from_folders(
    train_folder="data/CameraRGB",
    train_target_folder="data/CameraSeg",
    val_split=0.1,
    transform_kwargs=dict(image_size=(256, 256)),
    num_classes=21,
    batch_size=4,
)

model = SemanticSegmentation(
    backbone="mobilenetv3_large_100",
    head="fpn",
    num_classes=datamodule.num_classes,
)

trainer = flash.Trainer(max_epochs=3, gpus=torch.cuda.device_count())
# trainer = flash.Trainer(max_epochs=1, gpus=torch.cuda.device_count())

with mlflow.start_run():
    trainer.finetune(model, datamodule=datamodule, strategy="freeze")

    MODEL_CHECKPOINT_LOCAL_PATH = "semantic_segmentation_model.pt"
    trainer.save_checkpoint(MODEL_CHECKPOINT_LOCAL_PATH)

    # predictions = trainer.predict(model, datamodule=datamodule)
    # print(predictions)
    # TODO: visualize segmentation results on these images and log those as artifacts to MLflow

    MODEL_ARTIFACT_PATH = "segmentation_model"
    mlflow.pytorch.log_model(pytorch_model=model, artifact_path=MODEL_ARTIFACT_PATH)

    # Convert to ONNX
    MODEL_ONNX_ARTIFACT_PATH = "segmentation_model_onnx"
    MODEL_ONNX_LOCAL_PATH = "segmentation_model.onnx"

    X = torch.rand([1, 3, 256, 256])
    print(f"Input shape: {X.shape}")

    # Initialize model from saved checkpoints
    model = SemanticSegmentation(
        backbone="mobilenetv3_large_100",
        head="fpn",
        num_classes=21,
    ).load_from_checkpoint(MODEL_CHECKPOINT_LOCAL_PATH).eval()

    model.to_onnx(
        MODEL_ONNX_LOCAL_PATH,
        X,
        export_params=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )

    model_onnx = onnx.load(MODEL_ONNX_LOCAL_PATH)
    mlflow.onnx.log_model(model_onnx, artifact_path=MODEL_ONNX_ARTIFACT_PATH)
