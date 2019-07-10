import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from   torchvision import models, transforms, utils
from   torch.utils.data import Dataset,DataLoader
from   torch.autograd import Variable as V

from common.data                            import CoRL2017Dataset as Dataset
from ImitationLearning.network.CodevillaNet import ResNetReg

from config import Config
from config import Global

import common.pytorch as T
from common.utils import iter2time

from random import shuffle
import numpy as np
import cv2 as cv
import secrets
import math
import h5py
import os

# CUDA
device = torch.device('cuda:0')

# Settings
__global = Global()
__config = Config()

# Parameters
batch_size     = 120
n_epoch        = 150
framePerSecond =  10

class ResNetRegressionModel(object):
    """ Constructor """
    def __init__(self):
        # Configure
        self._config = Config()
        self.    net = None
        
        # Paths
        self. _cookPath = self._config.cookdPath
        self._modelPath = self._config.modelPath
        self._graphPath = self._config.graphPath

        # Nets
        self.net = None

        # Weight Loss
        self._weightLoss = torch.Tensor([__config.lambda_steer, 
                                         __config.lambda_gas  , 
                                         __config.lambda_brake]).float()

        # Optimizator
        self._optimizer = optim.Adam(   self.net.parameters(), 
                                        lr    = __config.adam_lrate, 
                                        betas =(__config.adam_beta_1, 
                                                __config.adam_beta_2)   )



    """ Building """
    def build(self):
        self.net = ResNetReg()
        self.net = self.net.float()
        self.net = self.net.apply(T.xavierInit)
        self.net = self.net.to(device)


    """ Train """
    def train(self):
        ite        =  0
        stepView   = 50
        weightLoss = self._weightLoss.cuda(device) 
        
        # List files
        path = self. _cookPath
        mode = 'Train'
        files = [os.path.join(path,f) for f in os.listdir(path) 
                                            if os.path.isfile(os.path.join(path,f)) 
                                                                    and mode in f]
        shuffle(files)
        trainFiles = files
        # Loop over the dataset multiple times
        for epoch in range(n_epoch):
            
            print("Epoch",epoch+1,"-----------------------------------")
            
            running_loss = 0.0
            for file in trainFiles:
                print("Read:",file)
                
                # Files loop
                for i, data in enumerate(DataLoader(Dataset(file),
                                                    shuffle     = True,
                                                    pin_memory  = True,
                                                    batch_size  = __config.batch_size,
                                                    num_workers = __global.num_workers), 0):
                    # get the inputs; data is a list of [frame, steer]
                    frame, steer = data

                    frame = frame.to(device)
                    steer = steer.to(device)

                    # zero the parameter gradients
                    self._optimizer.zero_grad()
                    outputs = self.net(frame)
                    
                    loss = T.weightedLoss(outputs, steer, weightLoss)
                    loss.backward()
                    self._optimizer.step()

                    ite = ite + 1

                    # print statistics
                    running_loss += loss.item()
                    if i % stepView == (stepView-1):   # print every stepView mini-batches
                        print(i+1,":\tloss =",running_loss/stepView,"\t\t",iter2time(ite))
                        running_loss = 0.0
 
