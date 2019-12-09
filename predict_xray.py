# Pneumonia_Classifer:  predict x-ray
# Richard Scott McNew
# A02077329
# CS 6600: Intelligent Systems

from __future__ import absolute_import, division, print_function, unicode_literals
import os
import shutil
import tempfile
import sys
from time import sleep
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMAGE_SIDE = 450 # We have to resize the dataset images to reduce memory consumption

# get the current working directory
current_dir = os.getcwd()

MODEL_PATH = os.path.join(current_dir, "pneumonia_classifier_model.h5")

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


# create an image generator for the uploaded images
def create_image_generator(dir_name):
    print("Creating image generator for images in directory: ", dir_name)
    image_generator = ImageDataGenerator(rescale=1./255)
    data_gen = image_generator.flow_from_directory(
            directory=str(dir_name), 
            target_size=(IMAGE_SIDE, IMAGE_SIDE),
            color_mode="grayscale",
            class_mode=None) # class_mode has to be None to indicate there are no data labels
    return data_gen
  
# Round the prediction to a human-readable interpretation
def interpret_prediction(prediction):
    print("Prediction is:  {}".format(prediction))
    pred_list = prediction[0].tolist()
    max_index = pred_list.index(max(pred_list))
    if max_index == 0:
        print("NORMAL")
    else:
        print("PNEUMONIA")

# print usage
def print_usage():
    print("predict_xray.py chest_xray_image")

# get the full path to the image file argument
def get_arg_path(image_file):
    if not os.path.exists(image_file): 
        return os.path.join(current_dir, image_file)

##### Main section #####
# get the chest x-ray path and file
arg_count = len(sys.argv) - 1
if arg_count != 1:
    print_usage()
else:
    model = download_trained_model_from_google_drive_shared()
    # create temporary directory
    temp_dir_name = tempfile.mkdtemp()
    print('Created temporary directory: ', temp_dir_name)
    # cd to temporary directory
    os.chdir(temp_dir_name)
    # make subdirectory for image files;  per the Keras documentation:  
    # "Please note that in case of class_mode None, the data still needs 
    # to reside in a subdirectory of directory for it to work correctly."
    image_path = os.path.join(temp_dir_name, "images")
    os.mkdir(image_path)
    os.chdir(image_path)
    # move the chest x-ray image to the temporary directory
    shutil.copy(get_arg_path(sys.argv[1]), image_path)
    # preprocess uploaded images
    image_gen = create_image_generator(temp_dir_name)
    # run prediction
    prediction = model.predict(image_gen)
    # interpret and print out result
    interpret_prediction(prediction)
    # return to starting directory
    os.chdir(current_dir)
    # clean-up temporary directory
    shutil.rmtree(temp_dir_name)


