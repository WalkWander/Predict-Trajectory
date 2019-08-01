# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 15:24:55 2019

@author: liu'yuan
"""
import torch
import torch.nn as nn
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.normal_(param.data, mean=0, std=0.01)   

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def train(model, input_data, label_data, optimizer, criterion, clip, batch_size, device, epoch, N_EPOCHS):
    
    model.train()
    
    epoch_loss = 0
 
    for i in range (0,(input_data.shape[0]//batch_size)*batch_size, batch_size):
        
        encoder_data = input_data[i:i+batch_size,:,:,:,:].clone() # b,len, c, h, w
        decoder_label = label_data[i:i+batch_size,:,:,:,:].clone()
        label = decoder_label.contiguous().view(-1)

        optimizer.zero_grad()

        if epoch < N_EPOCHS - (N_EPOCHS // 3):     
            output = model(encoder_data, decoder_label, 1)
        else:
            output = model(encoder_data, decoder_label, 0)
        output_array=output[0]
        output_array1 = output_array.contiguous().view(-1)
        
        loss = criterion(output_array1, label)
       
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        
        epoch_loss += loss.item()
    return epoch_loss / (i+1)

def evaluate(model, input_data, label_data, criterion, batch_size, device, epoch, N_EPOCHS):
    
    model.eval()
    
    epoch_loss = 0
    
    X = np.arange(0, 320, 1)
    Y = np.arange(0, 240, 1)
    X, Y = np.meshgrid(X, Y)
    
    with torch.no_grad():
        for i in range (0,input_data.shape[0]):
        
            encoder_data = input_data[i:i+batch_size,:,:,:,:].clone()
            decoder_label = label_data[i:i+batch_size,:,:,:,:].clone()
            label = decoder_label.contiguous().view(-1)
   
            output = model(encoder_data, decoder_label, 0)
            output_array=output[0]
            output_array1 = output_array.contiguous().view(-1)
        
            loss = criterion(output_array1, label)
            epoch_loss += loss.item()

        if epoch == N_EPOCHS-1:
            for j in range(0, label_data.shape[1]): 
                a = output_array[0,j,0,:,:].cpu().detach().numpy()
                b = label_data[i,j,0,:,:].cpu().detach().numpy()
                max_index1, max_index2 = np.where(a==np.max(a))
                print(max_index1, max_index2)
                b_max_index1, b_max_index2 = np.where(b==np.max(b))
                print(b_max_index1, b_max_index2)
                " show "
                fig = plt.figure()
                ax = Axes3D(fig)
                ax.plot_surface(X, Y, a, rstride=1, cstride=1, cmap='rainbow')
                plt.show()
                
                fig = plt.figure()
                bx = Axes3D(fig)
                bx.plot_surface(X, Y, b, rstride=1, cstride=1, cmap='rainbow')
                plt.show()
    
    return epoch_loss / (i+1)

def test_model(model, input_data, label_data):
    
    model.eval() 
    X = np.arange(0, 320, 1)
    Y = np.arange(0, 240, 1)
    X, Y = np.meshgrid(X, Y)
    predict_list = []
    label_list = []
    with torch.no_grad():
    
        for i in range (0,input_data.shape[0]):
            
            encoder_data = input_data[i:i+1,:,:,:,:].clone()
            decode_data = label_data[i:i+1,:,:,:,:].clone()
        
            decode_data[:,1:5,:,:,:] = decode_data[:,0:4,:,:,:]
            decode_data[:,0,:,:,:] = encoder_data[:,10,0,:,:]

            output = model(encoder_data, decode_data, 0)
            
            output_array=output[0]          
            
            for j in range(0, 5):
                a = output_array[0,j,0,:,:].cpu().detach().numpy()
                b = label_data[i,j,0,:,:].cpu().detach().numpy()
                max_index1, max_index2 = np.where(a==np.max(a))
#                print(max_index1, max_index2)
                b_max_index1, b_max_index2 = np.where(b==np.max(b))
#                print(b_max_index1, b_max_index2)
#                " show "
#                fig = plt.figure()
#                ax = Axes3D(fig)
#                ax.plot_surface(X, Y, a, rstride=1, cstride=1, cmap='rainbow')
#                plt.show()
#                
#                fig = plt.figure()
#                bx = Axes3D(fig)
#                bx.plot_surface(X, Y, b, rstride=1, cstride=1, cmap='rainbow')
#                plt.show()
                predict_list.append([max_index1, max_index2])
                label_list.append([b_max_index1, b_max_index2])
    
        predict_list = np.array(predict_list)
        label_list = np.array(label_list)

    return predict_list, label_list


