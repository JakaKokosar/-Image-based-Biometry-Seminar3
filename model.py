import os

from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Sequential, load_model as tf_load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, BatchNormalization, MaxPooling2D

from configs import IMAGE_HEIGHT, IMAGE_WIDTH, MODELS_PATH


def get_model() -> Sequential:

    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 1)))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def load_model(model_name: str) -> Sequential:
    return tf_load_model(os.path.join(MODELS_PATH, model_name))


# model = Sequential(
#     [
#         Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 1)),
#         MaxPooling2D(),
#         Conv2D(32, (3, 3), padding='same', activation='relu'),
#         MaxPooling2D(),
#         Conv2D(64, (3, 3), padding='same', activation='relu'),
#         MaxPooling2D(),
#         Flatten(),
#         Dropout(0.5),
#         Dense(512, activation='relu'),
#         Dense(7, activation='softmax'),
#     ]
# )
# model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
