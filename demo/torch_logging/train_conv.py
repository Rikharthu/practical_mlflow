# Train model with a conv as static feature extractor
import mlflow

from model import create_conv_model, train_model
from torch import nn, optim
from torch.optim import lr_scheduler
import os

os.environ["MLFLOW_TRACKING_URI"] = "http://192.168.0.97"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://192.168.0.97:9000"
os.environ["AWS_ACCESS_KEY_ID"] = "minio"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minio123"

OPTIMIZER = "SGD"
OPTIMIZER_LEARNING_RATE = 0.001
OPTIMIZER_MOMENTUM = 0.9
LEARNING_RATE_SCHEDULER = "StepLR"
LEARNING_RATE_STEP_SIZE = 7
LEARNING_RATE_GAMMA = 0.1
NUM_EPOCHS = 25

model = create_conv_model()

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer = optim.SGD(model.parameters(), lr=OPTIMIZER_LEARNING_RATE, momentum=OPTIMIZER_MOMENTUM)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=LEARNING_RATE_STEP_SIZE, gamma=LEARNING_RATE_GAMMA)

# Set an experiment
mlflow.set_experiment("pytorch_logging")

with mlflow.start_run() as run:
    # Log MLflow parameters
    mlflow.log_param("model_type", "conv_extractor")
    mlflow.log_param("optimizer", OPTIMIZER)
    mlflow.log_param("learning_rate", OPTIMIZER_LEARNING_RATE)
    mlflow.log_param("momentum", OPTIMIZER_MOMENTUM)
    mlflow.log_param("scheduler", LEARNING_RATE_SCHEDULER)
    mlflow.log_param("learning_rate_step_size", LEARNING_RATE_STEP_SIZE)
    mlflow.log_param("learning_rate_gamme", LEARNING_RATE_GAMMA)
    mlflow.log_param("num_epochs", NUM_EPOCHS)

    model = train_model(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=exp_lr_scheduler,
        num_epochs=NUM_EPOCHS
    )

    mlflow.pytorch.log_model(pytorch_model=model, artifact_path="model_ft")
