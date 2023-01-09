import numpy as np
from typing import Tuple, Dict, Optional
import tensorflow as tf
from tensorflow.data import AUTOTUNE
import pathlib
import os

IMAGE_SIZE = (128, 128)

GENDER_LABEL_TO_NAME: Dict[int, str] = {
    0: "male",
    1: "female"
}


def load_image(image_file_path, image_size: Tuple[int, int] = IMAGE_SIZE) -> Tuple[tf.Tensor, Dict[str, int]]:
    image = tf.io.read_file(image_file_path)

    image = tf.image.decode_jpeg(image)
    # For PNG images we would use tf.image.decode_png()

    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.image.resize(image, image_size)

    image_file_name = tf.strings.split(image_file_path, os.path.sep)[-1]
    # Remove extension
    image_file_name = tf.strings.split(image_file_name, ".")[0]

    labels = tf.strings.split(image_file_name, "_")
    age = tf.strings.to_number(labels[0], out_type=tf.dtypes.int32)
    gender = tf.strings.to_number(labels[1], out_type=tf.dtypes.int32)

    return (image, {"age": age, "gender": gender})


def create_utk_face_dataset(images_directory: str) -> tf.data.Dataset:
    """
    Creates a UTK-Face dataset from the specified images directory.
    This functions does not perform any additional configurations, such as batching,
    prefetching, augmentations, etc.
    """
    if not images_directory.endswith("/"):
        images_directory += "/"

    dataset = tf.data.Dataset.list_files(
        str(pathlib.Path(images_directory + "*.jpg")),
        shuffle=False,
    )
    # Set reshuffle_each_iteration to False to avoid getting same samples both in validation and test dataset
    dataset = dataset.shuffle(len(dataset), reshuffle_each_iteration=False)
    dataset = dataset.map(load_image, num_parallel_calls=AUTOTUNE)
    return dataset
