# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 14:24:46 2019

@author: liu'yuan
"""

import torch
import torch.optim as optim

import time
import gc
import matplotlib.pyplot as plt
#import scipy.io as io
#from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from model_sequential import Seq2Seq, Decode_ConvLSTM, ConvLSTM
from data_heatmap import trainging_data
from train_val import evaluate, train, init_weights, count_parameters, epoch_time


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.deterministic = True

"import trainging data" 
response_dataFile = 'KITTI_training0006a_heatmaps320_240.mat'
label_dataFile = 'KITTI_label0006a_320_240.mat'
batch_data1, batch_data_label1 = trainging_data(response_dataFile, label_dataFile)

batch_data = torch.from_numpy(batch_data1).float().to(device)
batch_data_label = torch.from_numpy(batch_data_label1).float().to(device)
del batch_data1,batch_data_label1
gc.collect()

input_data = batch_data[:,:,:,:,:]
label_data = batch_data_label[:,:,:,:,:]
print(input_data.shape) # b, len, c, h, w

" model "
input_size=(60, 80) #(height, width) (72, 96), (18, 24)
input_dim=32 #channels conv 16
hidden_dim=[32]
kernel_size=(3,3)
num_layers=1

enc = ConvLSTM(input_size, input_dim, hidden_dim, kernel_size, num_layers, batch_first=True, bias=True, return_all_layers=False)
dec = Decode_ConvLSTM(input_size, input_dim, hidden_dim, kernel_size, num_layers, batch_first=True, bias=True, return_all_layers=True)
model = Seq2Seq(enc, dec, device).to(device)

" init_weights "     
model.apply(init_weights)
print(f'The model has {count_parameters(model):,} trainable parameters')

" training "
criterion = torch.nn.L1Loss(reduction = 'mean') #MSELoss()  sum
optimizer = optim.Adam(model.parameters())
CLIP = 1
N_EPOCHS = 100
batch_size = 1
val_batch_size = 1
train_loss_list = []
valid_loss_list = []

for epoch in range(N_EPOCHS):
    
    start_time = time.time()

    train_loss = train(model, input_data, label_data, optimizer, criterion, CLIP, batch_size, device, epoch, N_EPOCHS)
    valid_loss = evaluate(model, input_data, label_data, criterion, val_batch_size, device, epoch, N_EPOCHS)
    
    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.6f}')
    print(f'\t Val. Loss: {valid_loss:.6f}')

    train_loss_list.append(train_loss)
    valid_loss_list.append(valid_loss)

torch.save(model.state_dict(), 'tut2-model.pt')

" plot "
x1 = range(1, N_EPOCHS+1)
x2 = range(1, N_EPOCHS+1)
y1 = train_loss_list
y2 = valid_loss_list
plt.plot(x1, y1, 'o-')
plt.plot(x1, y2, 'x-')
plt.title('Train accuracy vs. epoches')
plt.ylabel('train accuracy')
plt.show()



















