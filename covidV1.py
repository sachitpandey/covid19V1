# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 18:20:16 2020

@author: sachi
"""

#import csv 
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Model
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
import os
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import RMSprop

baseModel = VGG16(weights='imagenet', include_top=False,
	input_tensor=Input(shape=(256, 256, 3)))

for layer in baseModel.layers:
	layer.trainable = False
    
transfer_model = baseModel.output
transfer_model = AveragePooling2D(pool_size=(4, 4))(transfer_model)
transfer_model = Flatten()(transfer_model)
transfer_model = Dense(64, activation="relu")(transfer_model)
transfer_model = Dropout(0.2)(transfer_model)
transfer_model = Dense(1, activation="sigmoid")(transfer_model)

model = Model(inputs=baseModel.input, outputs=transfer_model)

model.compile(loss="binary_crossentropy", optimizer= RMSprop(lr=0.001),
	metrics=["accuracy"])

COVID_SOURCE_DIR = r'C:\Users\sachi\Desktop\keras-covid-19\dataset\covid'
NORMAL_SOURCE_DIR = r'C:\Users\sachi\Desktop\keras-covid-19\dataset\normal'
SOURCE_DIR = r'C:\Users\sachi\Desktop\keras-covid-19\dataset'
print(len(os.listdir(COVID_SOURCE_DIR)))
print(len(os.listdir(NORMAL_SOURCE_DIR)))

EPOCHS = int(input("Enter no. of epochs: "))
train_datagen = ImageDataGenerator(rescale=1./255,
    validation_split=0.2) # set validation split

train_generator = train_datagen.flow_from_directory(
    SOURCE_DIR,
    class_mode='binary',
    subset='training') # set as training data

validation_generator = train_datagen.flow_from_directory(
    SOURCE_DIR, # same directory as training data
    class_mode='binary',
    subset='validation') # set as validation data

H=model.fit_generator(
    train_generator,
    validation_data = validation_generator, 
    epochs = EPOCHS)

N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy on COVID-19 Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")