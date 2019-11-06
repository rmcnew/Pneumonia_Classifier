# Richard Scott McNew
# A02077329
# CS 6600: Intelligent Systems

import pathlib
import tempfile
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import IPython.display as display
from PIL import Image

# enable tensorflow AUTOTUNE
AUTOTUNE = tf.data.experimental.AUTOTUNE

BATCH_SIZE = 25
CLASS_NAMES = np.array(['NORMAL', 'PNEUMONIA'])
IMAGE_WIDTH = 1000
IMAGE_HEIGHT = 1000
SHUFFLE_SIZE = 100

# dataset paths
dataset = pathlib.Path("./dataset")
test = dataset.joinpath("test")
train = dataset.joinpath("train")
validate = dataset.joinpath("validate")

test_files = tf.data.Dataset.list_files(str(test/'*/*.jpeg'))
train_files = tf.data.Dataset.list_files(str(train/'*/*.jpeg'))
validate_files = tf.data.Dataset.list_files(str(validate/'*/*.jpeg'))

# temp file for dataset caching
def create_cache():
    temp_handle, temp_path = tempfile.mkstemp()
    return temp_path

def clean_cache(temp_path):
    tPath = pathlib.Path(temp_path)
    tPath.unlink(missing_ok=True)

# dataset loading and processing
def extract_label(path):
    parts = tf.strings.split(path, '/')
    return parts[-2]

def load_image_and_label(file_path):
    label = extract_label(file_path)
    image = tf.io.read_file(file_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [IMAGE_WIDTH, IMAGE_HEIGHT])
    return image, label

def load_dataset(dataset_files):
    labeled_dataset = dataset_files.map(load_image_and_label, num_parallel_calls=AUTOTUNE)
    return labeled_dataset

def load_train_dataset():
    return load_dataset(train_files)

def load_test_dataset():
    return load_dataset(test_files)

def load_validate_dataset():
    return load_dataset(validate_files)

def check_dataset(labeled_dataset, num_to_check):
    for image, label in labeled_dataset.take(num_to_check):
        print("Image shape: ", image.numpy().shape)
        print("Label: ", label.numpy())

# prepare dataset into batches and cache proprocessed data
def prepare_dataset(labeled_dataset):
    temp_path = create_cache() # need to save temp file path and clean up after use
    labeled_dataset = labeled_dataset.cache(temp_path)
    labeled_dataset = labeled_dataset.shuffle(buffer_size=SHUFFLE_SIZE)
    labeled_dataset = labeled_dataset.repeat()
    labeled_dataset = labeled_dataset.batch(BATCH_SIZE)
    labeled_dataset = labeled_dataset.prefetch(buffer_size=AUTOTUNE)
    return labeled_dataset, temp_path

def prepare_train_dataset():
    return prepare_dataset(load_train_dataset())

def prepare_test_dataset():
    return prepare_dataset(load_test_dataset())

def prepare_validate_dataset():
    return prepare_dataset(load_validate_dataet())




