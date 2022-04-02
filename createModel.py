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
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
warnings.filterwarnings('ignore')

def createModel(train_iterator, valid_iterator):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(256, 256, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    
    return model