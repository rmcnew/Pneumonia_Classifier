# Pneumonia_Classifer
# Richard Scott McNew
# A02077329
# CS 6600: Intelligent Systems

from __future__ import absolute_import, division, print_function, unicode_literals
import datetime
import io
import os
import pathlib
from time import sleep
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# enable accelerated linear algebra
tf.config.optimizer.set_jit(True)
# enable tensorflow AUTOTUNE
AUTOTUNE = tf.data.experimental.AUTOTUNE

################### Constants #######################
BATCH_SIZE = 8  # Use small batches to allow current batch to fit in GPU memory for faster training
IMAGE_SIDE = 450 # We have to resize the dataset images to reduce memory consumption
SHUFFLE_SIZE = 25 # We shuffle the images to ensure a better fit 
EPOCHS = 1  # Only run a few epochs at a time since Colaboratory times out without interactive use

# get the current working directory
current_dir = os.getcwd()

######### Dataset Download and Path Construction #####################
# Local path to the dataset
DATASET_PATH = os.path.join(current_dir, "dataset")

# There is a copy of the Pneumonia dataset in my Pneumonia_Classifier GitHub repo
# We can clone the 'dataset_only' branch to get a local copy
def get_dataset_files_from_github():
    if not os.path.isdir(DATASET_PATH):
        print("Downloading the dataset from GitHub . . .")
        os.system("git clone -b dataset_only https://github.com/rmcnew/Pneumonia_Classifier.git") 
    else:
        print("Using previously downloaded dataset")

# There is a tarball of the Pneumonia dataset available as a publicly shared link
# from my Google Drive account.  This is probably the fastest way to download a 
# local copy of the dataset since it should be all within Google's networks
def get_dataset_files_from_google_drive_shared():
    if not os.path.isdir(DATASET_PATH):
        print("Downloading the dataset from Google Drive shared link . . .")
        os.system("gdown https://drive.google.com/uc?id=1u2_Ap4rOxHuEKnSb5te070skuoXcJTX9")
        print("Download completed!  Untarring the dataset . . .")
        os.system("tar xjf Pneumonia_Classifier_dataset.tar.bz2")
        print("Dataset is ready!")
    else:
        print("Using previously downloaded dataset")

# Download a local copy of the dataset and then build paths 
# to the different dataset subsets: 'train', 'test', and 'validate'
#get_dataset_files_from_github()
get_dataset_files_from_google_drive_shared()
dataset = pathlib.Path(DATASET_PATH)
test = dataset.joinpath("test")
test_count = len(list(test.glob('**/*.jpeg')))
train = dataset.joinpath("train")
train_count = len(list(train.glob('**/*.jpeg')))
validate = dataset.joinpath("validate")
validate_count = len(list(validate.glob('**/*.jpeg')))


####################### Dataset Preprocessing #########################
# The train_image_generator applies random transformations to the 
# Train dataset subset to provide dataset augmentation since the dataset
# is small and we want to avoid overfitting
def create_train_image_generator():
    train_image_generator = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1, 
        zoom_range=0.2, 
        shear_range=0.05,
        fill_mode='nearest')
    train_data_gen = train_image_generator.flow_from_directory(
            batch_size=BATCH_SIZE, 
            directory=str(train), 
            shuffle=True, 
            target_size=(IMAGE_SIDE, IMAGE_SIDE),
            color_mode="grayscale",
            class_mode='categorical')
    return train_data_gen

# Prepare images in the Test dataset subset for use
def create_test_image_generator():
    test_image_generator = ImageDataGenerator(rescale=1./255)
    test_data_gen = test_image_generator.flow_from_directory(
            batch_size=BATCH_SIZE, 
            directory=str(test), 
            target_size=(IMAGE_SIDE, IMAGE_SIDE),
            color_mode="grayscale", 
            class_mode='categorical')
    return test_data_gen

# Prepare images in the Validate dataset subset for use
def create_validate_image_generator():
    validate_image_generator = ImageDataGenerator(rescale=1./255)
    validate_data_gen = validate_image_generator.flow_from_directory(
            batch_size=BATCH_SIZE, 
            directory=str(validate), 
            target_size=(IMAGE_SIDE, IMAGE_SIDE),
            color_mode="grayscale", 
            class_mode='categorical')
    return validate_data_gen


############################ Model Creation, Loading, and Saving ##############################
### This model tends to approach the upper limit of available memory of the Colaboratory virtual machine.
### Larger models with more layers or greater numbers of features run into "Out of Memory" errors
def create_model():
    model = Sequential([
        Conv2D(300, 10, padding='same', activation='relu', kernel_regularizer='l2', 
               input_shape=(IMAGE_SIDE, IMAGE_SIDE, 1)),
        Conv2D(300, 10, padding='same', activation='relu', kernel_regularizer='l2'),
        MaxPooling2D(10),
        Dropout(0.2),
        Conv2D(100, 4, padding='same', activation='relu', kernel_regularizer='l2'),
        Conv2D(100, 4, padding='same', activation='relu', kernel_regularizer='l2'),
        MaxPooling2D(),
        Dropout(0.2),
        Conv2D(50, 3, padding='same', activation='relu', kernel_regularizer='l2'),
        Conv2D(50, 3, padding='same', activation='relu', kernel_regularizer='l2'),
        MaxPooling2D(),
        Flatten(),
        Dense(300, activation='relu'),
        Dense(2, activation='softmax')
    ])
    model.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy']) 
    return model

MODEL_PATH = os.path.join(current_dir, "pneumonia_classifier_model.h5")

# save the model to disk
def save_model(model):
    model.save(MODEL_PATH, overwrite=True, include_optimizer=True, save_format='h5')

# load the model from disk
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH, custom_objects=None, compile=True)
    return model

###### Trained model from my Google Drive shared link #######
# There is the trained persisted model that is shared from my Google Drive account.
# This model can be downloaded by anyone who has the shared link URL.
# The 'gdown' command line tool is used to perform non-interactive downloads.
# This is the 'final' trained model that is trained to be as accurate as possible.

# Download a copy of the trained model to evaluate it against the Test or Validate
# dataset subsets or to predict using an uploaded chest X-ray image
def download_trained_model_from_google_drive_shared(download_anyway=False):
    if not os.path.exists(MODEL_PATH) or download_anyway:
        print("Downloading trained model from Google Drive shared link . . .")
        os.system("gdown https://drive.google.com/uc?id=1wOB-6Tn4-kexFliYaoO42Qvfneih81g7")
        print("Download completed!  Loading the model . . .")
    else:
        print("Using previously downloaded model")
    model = tf.keras.models.load_model(MODEL_PATH, custom_objects=None, compile=True)
    print("Model loaded.")
    return model


###################### Model Training and Evaluation ##############################
# Create the initial version of the model and run some training epochs
# After training epochs run, save the trained model to Google Drive or download it
def train_model():
    print("Training model from scratch")
    print("Preparing Train and Test dataset subsets")
    train_data_gen = create_train_image_generator()
    test_data_gen = create_test_image_generator()
    model = create_model()    
    print("Training model . . .")
    history = model.fit(
        train_data_gen,        
        steps_per_epoch=train_count // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=test_data_gen,        
        validation_steps=test_count // BATCH_SIZE
    ) 
    save_model(model)


# Load the previously trained model from Google Drive, run more training 
# epochs, and then save the more trained model back to Google Drive
def train_model_more():
    print("Training model more")
    print("Preparing training and testing datasets")
    train_data_gen = create_train_image_generator()
    test_data_gen = create_test_image_generator()
    model = load_model()    
    print("Training model more . . .")
    history = model.fit(
        train_data_gen,
        steps_per_epoch=train_count // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=test_data_gen,
        validation_steps=test_count // BATCH_SIZE
    )    
    save_model(model)
    
# Download the trained model from the Google Drive shared link
# and then evaluate the trained model's accuracy against the 
# Test dataset subset    
def test_trained_model():
    print("Running model against Test dataset subset")
    test_data_gen = create_test_image_generator()
    model = download_trained_model_from_google_drive_shared()
    model.evaluate(test_data_gen)

# Download the trained model from the Google Drive shared link
# and then evaluate the trained model's accuracy against the 
# Validate dataset subset
def validate_trained_model():
    print("Running model against Validate dataset subset")
    validate_data_gen = create_validate_image_generator()
    model = download_trained_model_from_google_drive_shared()
    model.evaluate(validate_data_gen)


############ Main Section ###############
# These function calls will need to be 
# commented out or uncommented depending 
# on what you want to do 

#train_model()            # <-- Run to train the model from scratch and save the trained model
#train_model_more()        # <-- Run to load a trained model from disk and train it more
#test_trained_model()     # <-- Run to see accuracy against Test dataset subset
validate_trained_model() # <-- Run to see accuracy against Validate dataset subset

