from asyncio.windows_events import NULL
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
from sklearn.model_selection import train_test_split

# local imports
import loadImages as li
import randomizeImage as ri
import createModel as cm

df = li.loadImages()

train, test = train_test_split(df, test_size=0.2, random_state=42)


train_iterator, valid_iterator = ri.randomizeImageTrain(train,test)
model = cm.createModel(train_iterator,valid_iterator) 

history = model.fit_generator(train_iterator, epochs=200, validation_data=valid_iterator)

## visualize the loss and accuracy of the model

# uploaded = files.upload()

# for fn in uploaded.keys():
 
#   # predicting images
#   path = '/content/' + fn
#   img = image.load_img(path, target_size=(128, 128))
#   x = image.img_to_array(img)
#   x /= 255
#   x = np.expand_dims(x, axis=0)

#   images = np.vstack([x])
#   classes = model.predict(images, batch_size=256)
#   print(classes[0])
#   if classes[0]>0.5:
#     print(fn + "is from the division 2")
#   else:
#     print(fn + "is from watch dogs 2")

