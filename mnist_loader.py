# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 09:56:17 2020

@author: ofekf
"""

import pickle
import gzip

import numpy as np

def load_data():
    #fuction preview
    f=gzip.open("mnist.pkl.gz", 'rb')
    traning_data, validation, test_data=pickle.load(f, encoding='latin1')
    f.close()
    return (traning_data, validation, test_data)
    
def load_data_wrapper():
    #fuction preview
    tr_d, ve_d, te_d = load_data()
    training_inputs=[np.reshape(x, (784,1)) for x in tr_d[0]]
    #training_results=[vectoraized_results(y) for y in tr_d[1]]
    validations_inputs=[np.reshape(x, (784,1)) for x in ve_d[0]]
    test_inputs=[np.reshape(x, (784,1)) for x in te_d[0]]
    #zipping
    training_data=list(zip(training_inputs,tr_d[1]))
    validation_data=list(zip(validations_inputs,ve_d[1]))
    test_data=list(zip(test_inputs,te_d[1]))
    
    return training_data,validation_data,test_data

def vectoraized_results(j):
    e=np.zeros((10,1))
    e[j]=1.0
    return e
    