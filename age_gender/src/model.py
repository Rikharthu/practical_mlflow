from tensorflow.keras.layers import Layer, Conv2D, MaxPooling2D, Flatten, Dense, Input, BatchNormalization, Activation, \
    Dropout, LeakyReLU
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.models import Model, Sequential
from typing import Tuple, Optional


class AgeGenderModel(Model):
    """
    Modular age classification and gender estimation model.
    """

    model: Model

    def __init__(
            self,
            input_shape: Tuple[int, int, int],
            augment: Optional[Layer] = None,
            backbone: Optional[Layer] = None,
            gender_head: Optional[Layer] = None,
            age_head: Optional[Layer] = None
    ):
        """
        :param input_shape:
        :param augment: image augmentation layers
        :param backbone: feature extractor
        :param gender_head: gender classification model head
        :param age_head: age estimation model head
        """
        super(AgeGenderModel, self).__init__()

        if backbone is None:
            backbone = self._build_backbone(input_shape)
        if gender_head is None:
            gender_head = self._build_gender_head()
        if age_head is None:
            age_head = self._build_age_head()

        input = Input(shape=input_shape, name="input")

        self.backbone = backbone
        self.gender_head = gender_head
        self.age_head = age_head
        self.augment = augment

        if self.augment is not None:
            x = self.augment(input)
            x = self.backbone(x)
        else:
            x = self.backbone(input)

        out_gender = self.gender_head(x)
        out_age = self.age_head(x)

        self.model = Model(
            inputs=input,
            outputs={
                "age": out_age,
                "gender": out_gender
            },
            name="age_gender_network"
        )

    def _build_gender_head(self) -> Layer:
        return Sequential([
            Flatten(),
            Dropout(0.8),

            Dense(512),
            BatchNormalization(),
            LeakyReLU(),
            Dropout(0.5),

            Dense(1, activation="sigmoid", name="gender")
        ], name="gender")

    def _build_age_head(self) -> Layer:
        return Sequential([
            Flatten(),
            Dropout(0.8),

            Dense(1024),
            BatchNormalization(),
            LeakyReLU(),
            Dropout(0.5),

            Dense(1024),
            BatchNormalization(),
            LeakyReLU(),
            Dropout(0.5),

            Dense(1, activation="linear", name="age")
        ], name="age")

    def _build_backbone(self, input_shape: Tuple[int, int, int]) -> Layer:
        initializer = HeNormal()

        return Sequential([
            Conv2D(32, (3, 3), padding="same", input_shape=input_shape, kernel_initializer=initializer),
            BatchNormalization(),
            LeakyReLU(),
            Conv2D(32, (3, 3), padding="same", input_shape=input_shape, kernel_initializer=initializer),
            BatchNormalization(),
            LeakyReLU(),
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            # Dropout(0.5),

            Conv2D(64, (3, 3), padding="same", kernel_initializer=initializer),
            BatchNormalization(),
            LeakyReLU(),
            Conv2D(64, (3, 3), padding="same", kernel_initializer=initializer),
            BatchNormalization(),
            LeakyReLU(),
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            # Dropout(0.5),

            Conv2D(128, (3, 3), padding="same", kernel_initializer=initializer),
            BatchNormalization(),
            LeakyReLU(),
            Conv2D(128, (3, 3), padding="same", kernel_initializer=initializer),
            BatchNormalization(),
            LeakyReLU(),
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            # Dropout(0.5),

            Conv2D(256, (3, 3), padding="same", kernel_initializer=initializer),
            BatchNormalization(),
            LeakyReLU(),
            Conv2D(256, (3, 3), padding="same", kernel_initializer=initializer),
            BatchNormalization(),
            LeakyReLU(),
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            # Dropout(0.5),

            Conv2D(256, (3, 3), padding="same", kernel_initializer=initializer),
            BatchNormalization(),
            LeakyReLU(),
            Conv2D(256, (3, 3), padding="same", kernel_initializer=initializer),
            BatchNormalization(),
            LeakyReLU(),
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        ], name="backbone")

    def call(self, inputs):
        # Forward call to inner model
        y_pred = self.model(inputs)
        return y_pred

    def summary(self):
        self.model.summary()
