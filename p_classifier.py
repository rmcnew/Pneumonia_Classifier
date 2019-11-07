# Richard Scott McNew
# A02077329
# CS 6600: Intelligent Systems

import pathlib
import tempfile
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# enable tensorflow AUTOTUNE
AUTOTUNE = tf.data.experimental.AUTOTUNE

BATCH_SIZE = 25
IMAGE_WIDTH = 250 
IMAGE_HEIGHT = 250 
SHUFFLE_SIZE = 25

# dataset paths
dataset = pathlib.Path("./dataset")
test = dataset.joinpath("test")
test_count = len(list(test.glob('**/*.jpeg')))
train = dataset.joinpath("train")
train_count = len(list(train.glob('**/*.jpeg')))
validate = dataset.joinpath("validate")
validate_count = len(list(validate.glob('**/*.jpeg')))

def create_train_image_generator():
    train_image_generator = ImageDataGenerator(rescale=1./255)
    train_data_gen = train_image_generator.flow_from_directory(
            batch_size=BATCH_SIZE, 
            directory=str(train), 
            shuffle=True, 
            target_size=(IMAGE_HEIGHT, IMAGE_WIDTH), 
            class_mode='binary')
    return train_data_gen

def create_test_image_generator():
    test_image_generator = ImageDataGenerator(rescale=1./255)
    test_data_gen = test_image_generator.flow_from_directory(
            batch_size=BATCH_SIZE, 
            directory=str(test), 
            target_size=(IMAGE_HEIGHT, IMAGE_WIDTH), 
            class_mode='binary')
    return test_data_gen

def create_validate_image_generator():
    validate_image_generator = ImageDataGenerator(rescale=1./255)
    validate_data_gen = validate_image_generator.flow_from_directory(
            batch_size=BATCH_SIZE, 
            directory=str(validate), 
            target_size=(IMAGE_HEIGHT, IMAGE_WIDTH), 
            class_mode='binary')
    return validate_data_gen


def create_model():
    model = Sequential([
        Conv2D(16, 4, padding='same', activation='relu', input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH ,3)),
        MaxPooling2D(),
        Conv2D(32, 4, padding='same', activation='relu'),
        MaxPooling2D(),
        Conv2D(64, 4, padding='same', activation='relu'),
        MaxPooling2D(),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(1, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) 
    return model

def train_model():
    train_data_gen = create_train_image_generator()
    test_data_gen = create_test_image_generator()
    model = create_model()
    history = model.fit_generator(
        train_data_gen,
        steps_per_epoch=train_count // BATCH_SIZE,
        epochs=30,
        validation_data=test_data_gen,
        validation_steps=test_count // BATCH_SIZE
    ) 
    model.save("pneumonia_classifier_model")


