import os

from pathlib import Path

ROOT_DIR = Path(__file__).parent
DATA_PATH = os.path.join(ROOT_DIR, 'data')
MODELS_PATH = os.path.join(ROOT_DIR, 'models')

ORIGINAL_DATA_PATH = os.path.join(ROOT_DIR, DATA_PATH, 'original')
TRAIN_DATA_PATH = os.path.join(ROOT_DIR, DATA_PATH, 'train')
VALIDATION_DATA_PATH = os.path.join(ROOT_DIR, DATA_PATH, 'val')
TEST_DATA_PATH = os.path.join(ROOT_DIR,  DATA_PATH, 'test')

EPOCHS = 50
BATCH_SIZE = 24
IMAGE_WIDTH, IMAGE_HEIGHT = 150, 150
