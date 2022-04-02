from cProfile import label
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import warnings
import os
import tqdm
import tensorflow as tf
import random
from keras.preprocessing.image import ImageDataGenerator
warnings.filterwarnings('ignore')

def randomizeImageTrain(train,test):
    ## Createing the data generator for the training
    train_image_generator = ImageDataGenerator(
        rescale=1./255, # Normalizing the pixel values
        rotation_range=40, # Randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.2, # Randomly shift images horizontally (fraction of total width)
        height_shift_range=0.2, # Randomly shift images vertically (fraction of total height)
        horizontal_flip=True, # Randomly flip images
        zoom_range=0.2, # Randomly zoom image
        fill_mode="nearest", # Fill mode
        shear_range=0.2 # Random shear
    )
    
    train_iterator = train_image_generator.flow_from_dataframe(train,
        x_col="input_path",
        y_col="label",
        target_size=(1024, 1024),
        batch_size=512,
        class_mode="binary"
    )

    valid_generator = ImageDataGenerator(rescale=1./255)

    valid_iterator = train_image_generator.flow_from_dataframe(test,
        x_col="input_path",
        y_col="label",
        target_size=(1024, 1024),
        batch_size=512,
        class_mode="binary",
        subset="validation"
    )

    return train_iterator , valid_iterator

