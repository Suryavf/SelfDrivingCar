import torch
import torch.nn as nn
from   torch.utils.data import Dataset,DataLoader

import numpy as np

from common.data import CoRL2017Dataset as Dataset
from common.utils import iter2time
from config import Config
from config import Global

# CUDA
device = torch.device('cuda:0')

# Settings
__global = Global()
__config = Config()

import numpy as np
import cv2 as cv
import secrets
import math
import h5py
import os


"""
Xavier initialization
"""
def xavierInit(model):
    if isinstance(model, nn.Linear):
        size = model.weight.size()
        fan_out = size[0] # number of rows
        fan_in  = size[1] # number of columns
        variance = np.sqrt(2.0/(fan_in + fan_out))
        model.weight.data.normal_(0.0, variance)


"""
Weighted loss
"""
# Weight to weightedLoss()
if __config.model in ['Basic', 'Multimodal', 'Codevilla18']:
    weightLoss = torch.Tensor([ __config.lambda_steer, 
                                __config.lambda_gas  , 
                                __config.lambda_brake]).float().cuda(device) 
if __config.model in ['Codevilla19']:
    weightLoss = torch.Tensor([ __config.lambda_steer * __config.lambda_action, 
                                __config.lambda_gas   * __config.lambda_action, 
                                __config.lambda_brake * __config.lambda_action,
                                __config.lambda_speed ]).float().cuda(device) 
def weightedLoss(input, target):
    loss = torch.abs(input - target)
    loss = torch.mean(loss,0)

    return torch.sum(loss*weightLoss)


"""
Metrics
"""
def MSE(input, target):
    loss = (input - target) ** 2
    return torch.mean(loss,0)


"""
Save checkpoint
"""
def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


"""
Train function
"""
def train(model,optimizer,lossFunc,files):
    stepView = __global.stepView
    global_iter = 0
    local_iter  = 0

    # Acomulative loss
    running_loss = 0.0
    
    # Train
    model.train()
    for file in files:
        print("Read:",file)

        # Files loop
        for i, data in enumerate(DataLoader(Dataset(file),
                                            pin_memory  = True,
                                            batch_size  = __config.batch_size,
                                            num_workers = __global.num_workers), 0):
            # get the inputs; data is a list of [frame, steer]
            frame, action = data

            frame  =  frame.to(device)
            action = action.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()
            outputs = model(frame)
            
            loss = lossFunc(outputs, action)
            loss.backward()
            optimizer.step()
            
            local_iter = global_iter + i # Update local iterator
            
            # Print statistics
            running_loss += loss.item()
            if i % stepView == (stepView-1):   # print every stepView mini-batches
                print(i+1,":\tloss =",running_loss/stepView,"\t\t",iter2time(local_iter))
                running_loss = 0.0

            # Bye loop 
            if local_iter>=__config.time_demostration:
                break
        
        # Bye loop 
        if local_iter>=__config.time_demostration:
            break
        global_iter = local_iter  # Update global iterator


"""
Validation function
"""
def validation(model,lossFunc,file):
    # Acomulative loss
    running_loss = 0.0
    count        = 0
    
    estimatedActions = list()

    # Metrics
    if __config.model in ['Basic', 'Multimodal', 'Codevilla18']:
        metrics = np.zeros((1,3))
    if __config.model in ['Codevilla19']:
        metrics = np.zeros((1,4))
    else:
        metrics = np.zeros((1,1))

    # Model to evaluation
    model.eval()
    
    # Data Loader
    with torch.no_grad():
        for i, data in enumerate(DataLoader(Dataset(file),
                                            pin_memory  = True,
                                            batch_size  = __config.batch_size,
                                            num_workers = __global.num_workers), 0):
            # get the inputs; data is a list of [frame, steer]
            frame, action = data
            frame  =  frame.to(device)
            action = action.to(device)

            # Forward pass
            output = model(frame)

            # Calculate the loss
            loss = lossFunc(output, action)
            running_loss += loss.item()

            # Mean squared error
            err = MSE(output,action)
            err = err.data.cpu().numpy()
            metrics += err

            estimatedActions.append( output.data.cpu().numpy() )

            # Update count
            count = count + 1 
    
    # Concatenate List
    estimatedActions = np.concatenate(estimatedActions,0)
    
    metrics      =      metrics/count
    running_loss = running_loss/count

    return running_loss,metrics
    