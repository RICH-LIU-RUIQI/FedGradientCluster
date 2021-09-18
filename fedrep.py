# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 14:34:27 2021

@author: csrqliu
"""

import math
import tensorflow as tf
import numpy as np
import logging
from rep_fun import *
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)


def fedrep(client_models, 
           server_model, 
           clients_datasets, 
           test_datasets, 
           data_distribution, 
           rounds,
           batch_size):
    
    Round = rounds
    num_client = len(client_models)
    client_scores = np.zeros(shape=[len(client_models), Round])
    BS = batch_size
    
    volume = 0
    client_volume = []
    average_scores = []
    # calculate the volume of clients' data
    for i in range(num_client):
        client_volume.append(len(clients_datasets[i][0]))
        volume += len(clients_datasets[i][0])
    
    for epoch in range(Round):
        print('------------ The %dth round ------------' %(epoch+1))
        
        # broadcasting
        for i in range(num_client):
            client_models[i] = set_weights_rep(client_model=client_models[i], 
                                           server_model=server_model)
        # local training $ local testing 
        
        print('local testing...')
        for i in range(num_client):
            # prepare the dataset
            te_dataset = tf_dataset(x=test_datasets[i][0],
                                 y=test_datasets[i][2],
                                 for_training=False)
            # local testing
            score = local_test(model=client_models[i],
                               dataset=te_dataset)
            client_scores[i][epoch] = score
            del te_dataset
            
        # average testing score 
        print('>>>>> average accuracy: %f <<<<<' %(np.mean(client_scores, axis=0)[epoch]))
        average_scores.append(np.mean(client_scores, axis=0)[epoch])
        
        print('local training...')
        for i in range(num_client):
            # get the batch for this round
            x1, y1, y2 = split_data(x=clients_datasets[i][0], 
                                    y1=clients_datasets[i][1],
                                    y2=clients_datasets[i][2], 
                                    distribution=data_distribution[i],
                                    batch_size=BS)
            del y1
            
            tr_dataset = tf_dataset(x1, y2, True)
            del x1, y2
            # train the head
            client_models[i] = local_train_many_round(model=client_models[i], 
                                                     dataset=tr_dataset,
                                                     update_head=True,
                                                     epochs=5)
            # train the feature extractor
            client_models[i] = local_train_many_round(model=client_models[i], 
                                                     dataset=tr_dataset,
                                                     update_head=False,
                                                     epochs=1)
            del tr_dataset
        
        # update global weights
        server_model = update_global_rep(client_models, server_model, volume, client_volume)
        
    return client_scores, average_scores