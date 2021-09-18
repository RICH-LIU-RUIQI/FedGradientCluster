# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 16:10:34 2021

@author: csrqliu
"""

import math
import tensorflow as tf
import numpy as np
import sklearn 
tf.autograph.set_verbosity(0)
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
import time

def l1_norm(x, y):
    return np.linalg.norm(x-y, ord=1)

def l2_norm(x, y):
    return np.linalg.norm(x-y, ord=2)

def distance_dot(x, y):
    return tf.tensordot(x, y, 1)

def obtain_grads(push_pull_values, p):
# =============================================================================
#     p is the hyperparameter
# =============================================================================
    client_gards = []
    pulls = []
    pushs = []
    with tf.device('/GPU:0'):
        # transfer the pushing and pulling
        for i in range(len(push_pull_values)):
            pulls.append(push_pull_values[i][0])
            pushs.append(push_pull_values[i][1])
        del push_pull_values
        # get new pushing and pulling for client
# =============================================================================
#         for client in range(len(pulls)):
#             for label in range(10):
#                 new_pulling = agg_pull(client, label, pulls, p)
#                 pulls[client][label] = new_pulling
#                 new_pushing = agg_push(client, label, pushs, p)
#                 pushs[client][label] = new_pushing
# =============================================================================
        # get grad for every client
        for i in range(len(pulls)):
            client_gards.append(pushs[i]-pulls[i])
        
    return client_gards

def agg_pull(client, label, pull_values, p):
    agg_pulling = np.zeros_like(pull_values[client][label])
    distances = np.zeros(shape=[1, len(pull_values)])
    client_pulls = pull_values[client][label]
    distance_sum = 0
    for client_index, client_pull in enumerate(pull_values):
        distance = distance_dot(client_pulls, client_pull[label].T)
        distance = math.pow(distance, -p)
        distances[:, client_index] = distance
        distance_sum += distance
    distances = distances / distance_sum
    agg_pulling = client_pulls
    for i in range(len(pull_values)):
        agg_pulling += distances[:, i] * pull_values[i][label].reshape(10, )
        
    return agg_pulling
        
def agg_push(client, label, pull_values, p):
    agg_pulling = np.zeros_like(pull_values[client][label])
    distances = np.zeros(shape=[1, len(pull_values)])
    client_pulls = pull_values[client][label]
    distance_sum = 0
    for client_index, client_pull in enumerate(pull_values):
        distance = distance_dot(client_pulls, client_pull[label].T)
        distance = math.pow(distance, -p)
        distances[:, client_index] = distance
        distance_sum += distance
    distances = distances / distance_sum
    agg_pulling = client_pulls
    for i in range(len(pull_values)):
        agg_pulling += distances[:, i] * pull_values[i][label].reshape(10, )
        
    return agg_pulling
    