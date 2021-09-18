# -*- coding: utf-8 -*-
"""
Created on Sat Sep 18 11:26:58 2021

@author: lenovo
"""

import math
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from cluster_grad import *
tf.autograph.set_verbosity(0)
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
import time


def split_data(x, y1, y2, distribution, batch_size):
# =============================================================================
#     y1 is 0~9, y2 is onehot encode
# =============================================================================
    client_indexs = []
    sum_size = 0
    for label in range(len(distribution)):
        index = np.argwhere(y1 == label)
        size = (distribution[label] / sum(distribution)) * batch_size
        client_indexs.append(index[np.random.randint(len(index), size=(int(size), 1))])
        sum_size += int(size)
    client_index = np.vstack((client_indexs[0], client_indexs[1]))
    for i in range(1, len(client_indexs)-1):
        client_index = np.vstack((client_index, client_indexs[i+1]))
    try:
        client_index = client_index.reshape(sum_size)
    except:
        client_index = client_index.reshape(batch_size-1)
    x1, y1, y2 = x[client_index], y1[client_index], y2[client_index]
    return x1, y1, y2

def tf_dataset(x, y, for_training=True):
    if for_training:
        BS = len(x)
    else:
        BS = 64
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.shuffle(buffer_size=1024).batch(BS)
    return dataset


def local_train_many_round(model, dataset, update_head, epochs):
    if update_head:
        
        for layer in model.layers[:-1]:
            layer.trainable = False
        model.layers[-1].trainable = True
    else:
        for layer in model.layers[:-1]:
            layer.trainable = True
        model.layers[-1].trainable = False
    with tf.device('/GPU:0'):
        model.fit(dataset, epochs=epochs, verbose=0)
    for layer in model.layers:
        layer.trainable = True
    return model


def local_grad_head(model, x, y):
    intermediate_layer_model = tf.keras.Model(inputs=model.input,
                                 outputs=model.layers[-2].output)
    values = []
    value_1 = np.zeros(shape=(64, 10))
    value_2 = np.zeros(shape=(64, 10))
    for label in range(10):
        label_index = np.argwhere(y == label)
        label_data = x[label_index]
        with tf.device('/GPU:0'):
                label_data = label_data.reshape(-1, 32, 32, 3)
                # get the possiblity
                output_p = model.predict_on_batch(label_data)
                input_h = intermediate_layer_model.predict_on_batch(label_data)
                for j in range(len(label_data)):
                    p_1 = output_p[j][label]
                    # get the input of clf
                    try:
                        input_h_1 = input_h[j].reshape(64, )
                        value_1[:, label] += (1-p_1) * input_h_1
                    except:
                        print('no!')
                        input_h_1 = input_h[j].reshape(64, )
                        value_1[:, label] += (1-p_1) * input_h_1
            
        # data with other labels 
        label_index = np.argwhere(y != label)
        label_data = x[label_index]
        with tf.device('/GPU:0'):
                label_data = label_data.reshape(-1, 32, 32, 3)
                # get the possiblity
                output_p = model.predict_on_batch(label_data)
                input_h = intermediate_layer_model.predict_on_batch(label_data)
                for j in range(len(label_data)):
                    p_1 = output_p[j][label]
                    # get the input of clf
                    try:
                        input_h_1 = input_h[j].reshape(64, )
                        value_2[:, label] += (p_1) * input_h_1
                    except:
                        print('no!')
                        input_h_1 = input_h[j].reshape(64, )
                        value_2[:, label] += (p_1) * input_h_1
                    
    values.append(value_1)
    values.append(value_2)
    return values


def local_test(model, dataset):  # ok
    with tf.device('/GPU:0'):
        score = model.evaluate(dataset, verbose=1)
    return score[1]

def set_weights_rep(client_model, server_model):
    # avoid adjusting the last layer
    for index_layer, layer in enumerate(client_model.layers[:-1]):
        # gain the weights from client and global
        base_weights = layer.get_weights()
        global_weights = server_model.layers[index_layer].get_weights()
        # update the weight
        if len(global_weights) == 0:
            continue
        else:
            base_weights = global_weights
            # update the layer
        client_model.layers[index_layer].set_weights(base_weights)
    return client_model

def update_global_rep(client_models, server_model, volume, client_volume):
    
    for i, layer in enumerate(server_model.layers[:-1]):  # update layers except the last layer
        layer_weights = layer.get_weights()
        if len(layer_weights) == 0:
            continue
        for j, gw in enumerate(layer_weights):  
            gw = gw * 0
            for k, model in enumerate(client_models):
                gw += (client_volume[k]/volume) * (model.layers[i].get_weights()[j])  # average
            layer_weights[j] = gw
        # update the layer
        server_model.layers[i].set_weights(layer_weights)  # update the weights of layer
    return server_model

