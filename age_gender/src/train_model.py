from data import create_utk_face_dataset, GENDER_LABEL_TO_NAME, IMAGE_SIZE
import math
from tensorflow import keras as keras
from keras.models import Sequential
from keras.layers import RandomZoom, RandomFlip, RandomRotation, RandomTranslation, Input, Flatten, Dropout, \
    Dense, BatchNormalization, LeakyReLU
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras_vggface.vggface import VGGFace
from model import AgeGenderModel
import time
import tensorflow as tf
import os
import mlflow


def main():
    # TODO: make configurable/get from MLFlow
    DATA_DIR = "./data/UTKFace"
    TRAIN_VAL_SPLIT = 0.8
    BATCH_SIZE = 32
    INPUT_SHAPE = (IMAGE_SIZE[0], IMAGE_SIZE[1], 3)

    dataset = create_utk_face_dataset(images_directory=DATA_DIR)

    dataset_size = len(dataset)
    print(f"Dataset size: {dataset_size}")
    train_size = int(math.floor(dataset_size * TRAIN_VAL_SPLIT))

    print(f"Training dataset size: {train_size}")
    train_dataset = dataset.take(train_size)

    val_dataset = dataset.skip(train_size)
    val_size = len(val_dataset)
    print(f"Validation dataset size: {val_size}")

    # TODO: move to own python source file
    # Faces in the original dataset are only slightly rotated, hence use small rotation angle in augmentations
    MAX_ROTATION_DEGREES = 30
    FILL_MODE = "nearest"

    augment = Sequential([
        Input(shape=INPUT_SHAPE),
        RandomFlip("horizontal"),
        RandomZoom(
            height_factor=(0.10, -0.10),
            width_factor=(0.10, -0.10),
            fill_mode=FILL_MODE
        ),
        RandomRotation(
            float(MAX_ROTATION_DEGREES) / 360.0,
            fill_mode=FILL_MODE
        ),
        RandomTranslation(
            height_factor=0.10,
            width_factor=0.10,
            fill_mode=FILL_MODE
        )
    ], name="augment")

    augment.summary()

    # TODO: move to own source python file: model.py
    base_model = VGGFace(
        model="vgg16",
        weights="vggface",
        include_top=False,
        input_shape=INPUT_SHAPE
    )
    base_model.summary()

    gender_head = Sequential([
        Flatten(),
        # No need for custom activation here, as the last convolution layer has ReLU activation
        Dropout(0.8),

        # Add a couple more fully-connected layers for age estimation
        Dense(256),
        BatchNormalization(),
        LeakyReLU(),
        Dropout(0.5),

        Dense(1, activation="sigmoid", name="gender")
    ], name="gender")

    age_head = Sequential([
        Flatten(),
        Dropout(0.8),

        # Add a couple more fully-connected layers for age estimation
        Dense(256),
        BatchNormalization(),
        LeakyReLU(),
        Dropout(0.5),

        Dense(1, activation="linear", name="age")
    ], name="age")

    model = AgeGenderModel(
        input_shape=INPUT_SHAPE,
        backbone=base_model,
        age_head=age_head,
        gender_head=gender_head,
        augment=augment
    )

    model.summary()

    # TODO: log & plot architecture to MLFLow

    # In first part of transfer learning do not retrain feature extractor
    model.backbone.trainable = False

    NUM_EPOCHS = 30
    LEARNING_RATE = 0.001
    LEARNING_RATE_DECAY = LEARNING_RATE / NUM_EPOCHS

    optimizer = Adam(
        learning_rate=LEARNING_RATE,
        decay=LEARNING_RATE_DECAY
    )

    loss = {
        "age": "mean_absolute_error",
        "gender": "binary_crossentropy"
    }
    loss_weights = {
        "age": 1.0,
        "gender": 20.0
    }

    # We can't just extract features from the whole dataset and feed it to only the gender and age heads
    # because we employ the transfer-learning
    model.model.compile(
        optimizer=optimizer,
        loss=loss,
        loss_weights=loss_weights,
        metrics={
            "age": "mae",
            "gender": "accuracy"
        }
    )

    ds_train = (train_dataset
                .shuffle(BATCH_SIZE * 100)
                .batch(BATCH_SIZE)
                .prefetch(tf.data.AUTOTUNE))
    ds_val = (val_dataset
              .batch(BATCH_SIZE)
              .prefetch(tf.data.AUTOTUNE))

    # Here we do not save model checkpoints yet

    start = time.time()
    history_1 = model.model.fit(
        x=ds_train,
        validation_data=ds_val,
        epochs=NUM_EPOCHS,
    )
    end = time.time()

    train_duration = end - start
    print(f"Classifier training took {train_duration} seconds")

    # Fine-tune whole network
    model.backbone.trainable = True
    # Number of trainable parameters must be much larger because backbone was unfrozen
    model.summary()
    NUM_EPOCHS = 20
    LEARNING_RATE = 0.00001
    LEARNING_RATE_DECAY = LEARNING_RATE / NUM_EPOCHS
    CHECKPOINTS_DIR = "./checkpoints"

    loss = {
        "age": "mean_absolute_error",
        "gender": "binary_crossentropy"
    }
    loss_weights = {
        "age": 1.0,
        "gender": 20.0
    }

    optimizer = Adam(
        learning_rate=LEARNING_RATE,
        decay=LEARNING_RATE_DECAY
    )

    model.model.compile(
        optimizer=optimizer,
        loss=loss,
        loss_weights=loss_weights,
        metrics={
            "age": "mae",
            "gender": "accuracy"
        }
    )

    ds_train = (train_dataset
                .shuffle(BATCH_SIZE * 100)
                .batch(BATCH_SIZE)
                .prefetch(tf.data.AUTOTUNE))
    ds_val = (val_dataset
              .batch(BATCH_SIZE)
              .prefetch(tf.data.AUTOTUNE))

    checkpoint_format = "checkpoint.epoch-{epoch:02d}.loss-{loss:.2f}.val_loss-{val_loss:.2f}.age_loss-{age_loss:.2f}.age_mae-{age_mae:.2f}.val_age_loss-{val_age_loss:.2f}.val_age_mae-{val_age_mae:.2f}.gender_loss-{gender_loss:.2f}.gender_accuracy-{gender_accuracy:.2f}.val_gender_loss-{val_gender_loss:.2f}.val_gender_accuracy-{val_gender_accuracy:.2f}.h5"
    checkpoint_filepath = os.path.join(CHECKPOINTS_DIR, checkpoint_format)

    # TODO: add a checkpointthat would delete old checkpoints (keep top 3 on val_loss)
    #   https://github.com/tensorflow/tensorflow/issues/30695

    callbacks = [
        ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_best_only=True,
            monitor="val_loss",
            mode="min",
            save_freq="epoch",
            save_weights_only=False
        )
    ]

    start = time.time()
    history_2 = model.model.fit(
        x=ds_train,
        validation_data=ds_val,
        epochs=NUM_EPOCHS,
        callbacks=callbacks
    )
    end = time.time()

    train_duration_ft = end - start
    print(f"Fine-tuning took {train_duration} seconds")


if __name__ == "__main__":
    main()
