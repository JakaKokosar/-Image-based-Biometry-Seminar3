from tensorflow.keras.preprocessing.image import ImageDataGenerator

from configs import IMAGE_HEIGHT, IMAGE_WIDTH, BATCH_SIZE, TEST_DATA_PATH, TRAIN_DATA_PATH, VALIDATION_DATA_PATH

data_generator = ImageDataGenerator(rescale=1./255)

augmented_data_generator = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
    rotation_range=120,
    width_shift_range=.2,
    height_shift_range=.2,
    zoom_range=0.2,
    fill_mode='nearest'
)

train_data_generator = augmented_data_generator.flow_from_directory(
    TRAIN_DATA_PATH,
    shuffle=True,
    target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    color_mode='grayscale',
)

validation_data_generator = data_generator.flow_from_directory(
    VALIDATION_DATA_PATH,
    shuffle=True,
    target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    color_mode='grayscale',
)

test_data_generator = data_generator.flow_from_directory(
    TEST_DATA_PATH,
    target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode=None,
    shuffle=False,
    color_mode='grayscale'
)
