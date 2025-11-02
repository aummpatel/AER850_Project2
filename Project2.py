# Step 0: Import all required packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator as idg
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import os
from pathlib import Path

#---------------------------------------------------------------------------------------------
# Step 1: Data Processing and Pre-Processing

# Defining Input Image Shape (500,500,3) and batch size
img_h = 500
img_w = 500
img_channel = 3 #Question, why 3 even though images are all Grayscale
img_shape = (img_h,img_w,img_channel)
batch = 32

# Relative Directory Paths
# A. Find the directory where the current script file is saved
project_directory = os.path.dirname(os.path.realpath(__file__))
# Project_Directory = Path(__file__).resolve().parent

# B. Navigate to the 'Data' directory relative to the Project Directory
data_directory = os.path.join(project_directory, 'Project 2 Data','Data')
# Data_Directory = Project_Directory/'Project 2 Data'/'Data'

# C. Define Directory Paths for Training and Validation Data Relative to Project Directory
train_directory = os.path.join(data_directory, 'train')
valid_directory = os.path.join(data_directory, 'valid')
# train_directory = data_folder/'train'
# valid_directory = data_folder/'valid'

# Or simply 
# train_directory = 'Project 2 Data/Data/train'
# valid_directory = 'Project 2 Data/Data/valid'

#Question, which of the above three is meant by 'relative path'

#--------------------------------------------------------------------------------------------------------------------------------------
# Data Augmentation on training image set using ImageDataGenerator
train_datagen = idg(
    rotation_range = 10, #random rotation in the range +-10 degrees
    zoom_range = 0.15, #zoom in/out by +-15%
    shear_range = 0.2, #shear angle in degrees
    brightness_range = [0.85,1.15], #a +-15% variation in brightness 
    rescale = 1./255, #normalize RGB pixels from [0,255] into [0,1]
    horizontal_flip = True) #flip images horizontally

#validation images only rescaled since those would be used to validate our training effectiveness so augmentation is unnecessary
valid_datagen = idg(
    rescale = 1./255)

# Data Generator using ImageDataGenerator
train_gen = train_datagen.flow_from_directory(
    train_directory,
    target_size=(img_h,img_w),
    color_mode = 'rgb',
    classes = ['crack','missing-head','paint-off'],
    class_mode = 'categorical',
    batch_size = batch,
    shuffle = True,
    seed = 42)

valid_gen = valid_datagen.flow_from_directory(
    valid_directory,
    target_size=(img_h,img_w),
    color_mode = 'rgb',
    classes = ['crack','missing-head','paint-off'],
    class_mode = 'categorical',
    batch_size = batch,
    shuffle = True,
    seed = 42)

#--------------------------------------------------------------------------------------------------------------------------------------
# Generator using Keras’s built-in imagedatasetfromdirectory function
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_directory,
    labels = "inferred",
    label_mode = "categorical",
    class_names = ['crack','missing-head','paint-off'],
    batch_size = batch,
    image_size = (img_h,img_w),
    color_mode = "rgb",
    shuffle = False,
    verbose = True)

valid_ds = tf.keras.preprocessing.image_dataset_from_directory(
    valid_directory,
    labels = "inferred",
    label_mode = "categorical",
    class_names = ['crack','missing-head','paint-off'],
    batch_size = batch,
    image_size = (img_h,img_w),
    color_mode = "rgb",
    shuffle = False,
    verbose = True)

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.028,interpolation='nearest'), #same rotation of about 10deg
    layers.Rescaling(1./255),
    layers.RandomBrightness(0.15,value_range=[0.0,1.0]),
    layers.RandomZoom(height_factor=0.15,width_factor=0.15)
    ])

# Question is how do I augment the train dataset? This returns a tf.data.Dataset object, how to augment that?
# Make a separate augmentation keras.layers pipeline with preprocessing layers?
# For the above implementation no Shear available
# Which way to go with, depracated way or the modern way?

#--------------------------------------------------------------------------------------------------------------------------------------









