# -*- coding: utf-8 -*-
"""
Created on Sat Sep 18 10:59:34 2021

generalize dataset for FL

@author: lenovo
"""


# generalize dataset for clients

import tensorflow as tf
import numpy as np
import tensorflow.keras as tf_keras
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')  # 把讨厌的warning去掉
import os
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
####################### load data and split it #######################

def main_data(draw=False):
    cifar10 = tf.keras.datasets.cifar10
    (train_x, train_y), (test_x, test_y) = cifar10.load_data()
    train_x = np.expand_dims(train_x, -1)
    test_x = np.expand_dims(test_x, -1)
    train_y = train_y.reshape(50000,)
    test_y = test_y.reshape(10000,)
    net_type = 'cnn'
    
    
    # load client dataset
    clients_datasets = []
    data_distributions = []
    for i in range(8):
        dis = np.random.randint(1, 8, size=(1, 10)).tolist()[0]
        data_distributions.append(dis)
    for distribution in data_distributions:
        clients_datasets.append(load_client_data(train_x, train_y, 
                                                 net_type, sizes=distribution))
    if 1:
        plt.figure(figsize=(20, 6))

        plt.subplot(2, 4, 1)
        plt.bar(x=np.arange(10), height=data_distributions[0])
        plt.xticks(np.arange(10))
        plt.yticks(np.arange(8))
        
        plt.subplot(2, 4, 2)
        plt.bar(x=np.arange(10), height=data_distributions[1])
        plt.xticks(np.arange(10))
        plt.yticks(np.arange(8))
        
        plt.subplot(2, 4, 3)
        plt.bar(x=np.arange(10), height=data_distributions[2])
        plt.xticks(np.arange(10))
        plt.yticks(np.arange(8))
        
        plt.subplot(2, 4, 4)
        plt.bar(x=np.arange(10), height=data_distributions[3])
        plt.xticks(np.arange(10))
        plt.yticks(np.arange(8))
        
        plt.subplot(2, 4, 5)
        plt.bar(x=np.arange(10), height=data_distributions[4])
        plt.xticks(np.arange(10))
        plt.yticks(np.arange(8))
        
        plt.subplot(2, 4, 6)
        plt.bar(x=np.arange(10), height=data_distributions[5])
        plt.xticks(np.arange(10))
        plt.yticks(np.arange(8))
        
        plt.subplot(2, 4, 7)
        plt.bar(x=np.arange(10), height=data_distributions[6])
        plt.xticks(np.arange(10))
        plt.yticks(np.arange(8))
        
        plt.subplot(2, 4, 8)
        plt.bar(x=np.arange(10), height=data_distributions[7])
        plt.xticks(np.arange(10))
        plt.yticks(np.arange(8))
        
        plt.title('Data Distribution in Clients')
        plt.show();
        
    # load test dataset
    test_datasets = []
    test_distributions = data_distributions
    for distribution in test_distributions:
        test_datasets.append(load_test_data(test_x, test_y, net_type, sizes=distribution))
        
    return clients_datasets, test_datasets, data_distributions

def load_test_data(x, y, net_type, sizes, labels=range(10)):
    Dataset = []
    client_indexs = []
    i = 0
    test_y = y
    test_x = x
    for label in labels:
        size = int(sizes[i] * 50)
        index = np.argwhere(test_y == label)
        client_indexs.append(index[np.random.randint(len(index), size=(int(size), 1))])
        i += 1
    client_index = np.vstack((client_indexs[0], client_indexs[1]))
    for i in range(1, len(client_indexs)-1):
        client_index = np.vstack((client_index, client_indexs[i+1]))
    client_index = client_index.reshape(int(sum(sizes)*50))
    client1_train_x, client1_train_y = test_x[client_index], test_y[client_index]
    if net_type == 'dnn':
        client1_train_x = client1_train_x.reshape(-1, 32*32*3)
    else:
        client1_train_x = client1_train_x.reshape(-1, 32, 32, 3)
    client1_train_x = client1_train_x / 255.0
    train_y = tf.keras.utils.to_categorical(client1_train_y, 10)
    Dataset.append(client1_train_x)
    Dataset.append(client1_train_y)
    Dataset.append(train_y)
    return Dataset

def load_client_data(x, y, net_type, sizes, labels=range(10)):
    Dataset = []
    client_indexs = []
    i = 0
    train_y = y
    train_x = x
    for label in labels:
        size = int(sizes[i] * 15)
        index = np.argwhere(train_y == label)
        client_indexs.append(index[np.random.randint(low=0, high=len(index), size=(int(size), 1))])
        i += 1
    client_index = np.vstack((client_indexs[0], client_indexs[1]))
    for i in range(1, len(client_indexs)-1):
        client_index = np.vstack((client_index, client_indexs[i+1]))
    client_index = client_index.reshape(int(sum(sizes))*15)
    client1_train_x, client1_train_y = train_x[client_index], train_y[client_index]
    if net_type == 'dnn':
        client1_train_x = client1_train_x.reshape(-1, 32*32*3)
    else:
        client1_train_x = client1_train_x.reshape(-1, 32, 32, 3)
    client1_train_x = client1_train_x / 255.0
    train_y = tf.keras.utils.to_categorical(client1_train_y, 10)
    Dataset.append(client1_train_x)
    Dataset.append(client1_train_y)
    Dataset.append(train_y)
    return Dataset