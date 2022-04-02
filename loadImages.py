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


def loadImages():
    input_path = []
    labe = []

    ## Loading the data
    for type in os.listdir("Images"):
        for path in os.listdir("Images/"+type):
            if(type == "the_division2"):
                labe.append(1)
            else:
                labe.append(0)
            input_path.append("PetImages/"+type+"/"+path)

    ## Creating a dataframe
    df = pd.DataFrame()
    df['input_path'] = input_path
    df['label'] = labe
    df['label'] = df['label'].astype('str')
    df = df.sample(frac=1).reset_index(drop=True)

    return df

