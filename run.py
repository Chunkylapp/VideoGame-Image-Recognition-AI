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

history = model.fit_generator(train_iterator, epochs=50, validation_data=valid_iterator)

## visualize the loss and accuracy of the model

print(history.history)
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']




epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.plot(epochs, loss, 'r--', label='Training Loss')
plt.plot(epochs, val_loss, 'b--', label='Validation Loss')
plt.title('Training and Validation Accuracy and Loss')
plt.legend()
plt.figure()
plt.show()

