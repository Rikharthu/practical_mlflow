import torch
import flash
from flash.core.data.utils import download_data
from flash.text import TextClassificationData, TextClassifier
import mlflow

# %%
# Set this if running non-locally
mlflow.set_tracking_uri('http://localhost')

EXPERIMENT_NAME = "dl_model_chapter02"
mlflow.set_experiment(EXPERIMENT_NAME)

# Enable automatic PyTorch logging in MLFlow
# This will allow the default parameters, metrics, and model to be automatically
# logged to the MLflow tracking server.
mlflow.pytorch.autolog()

experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
print("experiment_id:", experiment.experiment_id)

# %%
download_data("https://pl-flash-data.s3.amazonaws.com/imdb.zip", "./data/")

datamodule = TextClassificationData.from_csv(
    input_field="review",
    target_fields="sentiment",
    train_file="data/imdb/train.csv",
    val_file="data/imdb/valid.csv",
    test_file="data/imdb/test.csv",
    batch_size=32
)

# %%
gpu_count = torch.cuda.device_count()
print(f"Number of GPU devices: {gpu_count}")

# https://huggingface.co/prajjwal1/bert-tiny
classifier_model = TextClassifier(backbone="prajjwal1/bert-tiny", num_classes=datamodule.num_classes)
trainer = flash.Trainer(max_epochs=10, gpus=gpu_count)

with mlflow.start_run(experiment_id=experiment.experiment_id, run_name="chapter02"):
    trainer.finetune(classifier_model, datamodule=datamodule, strategy="freeze")
    trainer.test(datamodule=datamodule)

# %% Predictions
print('Get prediction outputs for two sample sentences')
predict_module = TextClassificationData.from_lists(
    predict_data=[
        "Best movie I have seen.",
        "What a movie!",
    ],
    batch_size=2
)

predictions = trainer.predict(classifier_model, datamodule=predict_module)
print(predictions)
