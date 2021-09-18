# -*- coding: utf-8 -*-
"""
Created on Sat Sep 18 11:20:24 2021

@author: Rich Liu

build net for clients and server
"""
import tensorflow as tf
import numpy as np
import math
from tensorflow import keras

def build_model():
    lenet_5_model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16, kernel_size=5, strides=1,  activation='relu', 
                               input_shape=(32, 32, 3), padding='same'), #C1
        tf.keras.layers.AveragePooling2D(), #S2
        tf.keras.layers.Conv2D(32, kernel_size=5, strides=1, activation='relu', 
                               padding='valid'), #C3
        tf.keras.layers.AveragePooling2D(), #S4
        tf.keras.layers.Flatten(), #Flatten
        tf.keras.layers.Dense(512, activation='relu'), #C5
        tf.keras.layers.Dense(64, activation='relu'), #F6
        tf.keras.layers.Dense(10, activation='softmax', use_bias=False) #Output layer
    ])
    lenet_5_model.compile(loss='categorical_crossentropy',
                optimizer=tf.keras.optimizers.SGD(learning_rate=0.015, momentum=0.9, decay=1e-3),
                 metrics=['acc'])
    return lenet_5_model

def get_client_models():
    client_models = []
    for i in range(8):
        client_models.append(build_model())
    return client_models

def get_server_model():
    return build_model()

