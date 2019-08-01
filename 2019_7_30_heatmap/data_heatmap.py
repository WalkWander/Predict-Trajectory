# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 15:26:26 2019

@author: liu'yuan
"""
import scipy.io as scio
import numpy as np
import gc

def trainging_data(response_dataFile, label_dataFile):

    response_data = scio.loadmat(response_dataFile)
    response_data_array = np.array(response_data['heatmaps'])
    label_data = scio.loadmat(label_dataFile)
    label_data_array = np.array(label_data['label_data'])

    response_data_array = response_data_array.astype(np.float)
    label_data_array = label_data_array.astype(np.float)
    skip = 25 #25
    dim = int((label_data_array.shape[0]-22)//skip)
#    skip = 1
#    dim = int((label_data_array.shape[0]-46)//skip)
    batch_data = np.zeros((dim, 11, response_data_array.shape[1], response_data_array.shape[2])) # b,t,h,w
    batch_data_label = np.zeros((dim, 5, label_data_array.shape[1], label_data_array.shape[2]))

    batch_data_orig = np.zeros((11,response_data_array.shape[1], response_data_array.shape[2]))
    for i in range(0, dim):
        for j in range(0,11):
            batch_data_orig[j:j+1, :, :] = response_data_array[skip*i+4*j+1:skip*i+4*j+2, :, :]
            
        batch_data_3dim = batch_data_orig[np.newaxis,:,:] 
        batch_data[i:i+1,:,:,:] = batch_data_3dim
        
        label_data_orig = label_data_array[skip*i+42:skip*i+47, :] #+16
        label_data_3dim = label_data_orig[np.newaxis,:,:] 
        batch_data_label[i:i+1,:,:,:] = label_data_3dim
    
    
    batch_data = batch_data[:,:,np.newaxis,:,:] 
    batch_data_label = batch_data_label[:,:,np.newaxis,:,:] # b,t c, h, w

    batch_data = np.multiply(batch_data,1000000)
    batch_data_label = np.multiply(batch_data_label,100000)
    
    del label_data_3dim, batch_data_3dim, label_data_array, response_data_array, label_data_orig, batch_data_orig
    gc.collect()
    return batch_data, batch_data_label
