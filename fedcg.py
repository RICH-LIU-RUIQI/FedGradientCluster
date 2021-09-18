# -*- coding: utf-8 -*-
"""
Created on Sat Sep 18 12:24:02 2021

@author: lenovo
"""

import math
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from cluster_grad import *
from cg_fun import *
tf.autograph.set_verbosity(0)
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
import time

def fedcg(client_models, 
          server_model, 
          clients_datasets, 
          test_datasets, 
          p, 
          data_distribution,
          lr,
          batch_size,
          rounds):
    
    lr = lr
    Round = rounds
    train_head_epochs = 5
    num_client = len(client_models)
    client_scores = np.zeros(shape=[len(client_models), Round])
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    BS = batch_size
    
    volume = 0
    client_volume = []
    average_scores = []
    grad_values = []
    # calculate the volume of clients' data
    for i in range(num_client):
        client_volume.append(len(clients_datasets[i][0]))
        volume += len(clients_datasets[i][0])
    
    for epoch in range(Round):
        print('------------ The %dth round ------------' %(epoch+1))
        
        ############################ broadcasting #############################
        
        # broadcasting
        print('broadcasting the feature extractor...')
        for i in range(num_client):
            client_models[i] = set_weights_rep(client_model=client_models[i], 
                                           server_model=server_model)
            
        ########################## local testing #############################
        print('local testing...')
        for i in range(num_client):
            te_dataset = tf_dataset(test_datasets[i][0], 
                                    test_datasets[i][2],
                                    False)
            
            score = local_test(client_models[i], te_dataset)
            del te_dataset
            client_scores[i][epoch] = score
        print('>>>>> average accuracy: %f <<<<<' %(np.mean(client_scores, axis=0)[epoch]))
        average_scores.append(np.mean(client_scores, axis=0)[epoch])

        ##################### prepare the batch for clients #####################
        
        # get the batch for this round
        train_head_x = []
        train_head_y = []
        train_fe_dataset = []
        for i in range(num_client): 
            x1, y1, y2 = split_data(x=clients_datasets[i][0], 
                                    y1=clients_datasets[i][1],
                                    y2=clients_datasets[i][2], 
                                    distribution=data_distribution[i],
                                    batch_size=BS)
            train_head_x.append(x1)
            train_head_y.append(y1)
            train_fe_dataset.append(tf_dataset(x1, y2, True))
            
        
        print('local training on head...')
        for j in (range(train_head_epochs)):
            push_pull_values = []
            for i in range(num_client):
                # values 1 -> pull, values 2 -> push
                values = local_grad_head(model=client_models[i], 
                                                         x=train_head_x[i], 
                                                         y=train_head_y[i])
                push_pull_values.append(values)

            client_grads = obtain_grads(push_pull_values, p)
            del push_pull_values
            # update the parameter
            for i in range(num_client):
                tensor_grad = []
                grad = client_grads[i]
                if grad.shape != (64, 10):
                    print('the gradient shape is wrong', grad.shape)
                    return 0
                # tranform grad(numpy) to grad(tensor)
                tensor_grad.append(tf.convert_to_tensor(grad, dtype=tf.float32))
                opt.apply_gradients(zip(tensor_grad, 
                                        client_models[i].layers[-1].trainable_weights))
            
        print('local training on the feature extractor...')
        for i in range(num_client):
# =============================================================================
#             for j in (range(train_head_epochs)):
#                 dataset = tf.data.Dataset.from_tensor_slices((clients_datasets[i][0], clients_datasets[i][2])).batch(len(clients_datasets[i][0]))
#                 for (x,y) in dataset:
#                     with tf.device('/GPU:0'):
#                         with tf.GradientTape() as tape:
#                             pred = client_models[i](x, training=True)
#                             loss = tf.keras.losses.categorical_crossentropy(y_true=y, y_pred=pred)
#                             print(tf.reduce_mean(loss))
#                         grads1 = tape.gradient(loss, client_models[i].layers[-1].trainable_variables)
#                         opt.apply_gradients(zip(grads1, client_models[i].layers[-1].trainable_weights))
#             print('<-------->')
# =============================================================================
            
            # train on the feature extractor
            client_models[i] = local_train_many_round(client_models[i],
                                   train_fe_dataset[i],
                                   False,
                                   1)
        del train_head_x, train_head_y, train_fe_dataset
        # update global weights
        server_model = update_global_rep(client_models, server_model, volume, client_volume)
        
    return client_scores, average_scores