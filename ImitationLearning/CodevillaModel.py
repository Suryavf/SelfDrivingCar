import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from   torchvision import models, transforms, utils
from   torch.utils.data import Dataset,DataLoader
from   torch.autograd import Variable as V

from common.data                            import CoRL2017Dataset as Dataset
from ImitationLearning.network.CodevillaNet import ResNetReg
from config                                 import Config


import numpy as np
import cv2 as cv
import secrets
import math
import h5py
import os

# CUDA :D
device = torch.device('cuda:0')

# Parameters
batch_size     = 120
n_epoch        = 150
framePerSecond =  10


def timeExecution(ite):
    time = ite*batch_size/framePerSecond

    # Hours
    hour = np.floor(time/3600)
    time = time - hour*3600

    # Minutes
    minute = np.floor(time/60)
    time = time - minute*60

    # Seconds
    second = time

    # Text
    txt = ""
    if(  hour>0): txt = txt + str( hour ) + "h\t"
    else        : txt = txt + "\t"

    if(minute>0): txt = txt + str(minute) + "m\t"
    else        : txt = txt + "\t"

    return txt + str(second) + "s"
 

def weights_init(m):
    if isinstance(m, nn.Linear):
        size = m.weight.size()
        fan_out = size[0] # number of rows
        fan_in  = size[1] # number of columns
        variance = np.sqrt(2.0/(fan_in + fan_out))
        m.weight.data.normal_(0.0, variance)

def weightedLoss(input, target, w):
    loss = torch.abs(input - target)
    loss = torch.mean(loss,0)

    return torch.sum(loss*w)

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

    """ Building """
    def build(self):
        self.net = ResNetReg()
        self.net = net.float()
        self.net = net.apply(weights_init)
        self.net = net.to(device)

    """ Train """
    def train(self):
        ite        =  0
        stepView   = 50
        weightLoss = torch.Tensor([0.45 , 0.45 , 0.05]).float().cuda(device) 
        
        optimizer = optim.Adam(self.net.parameters(), lr=0.0002, betas=(0.7, 0.85))

        # List files
        path = self. _cookPath
        mode = 'Train'
        files = [os.path.join(path,f) for f in os.listdir(path) 
                                            if os.path.isfile(os.path.join(path,f)) 
                                                                    and mode in f]
        trainFiles = files
        # Loop over the dataset multiple times
        for epoch in range(n_epoch):
            running_loss = 0.0
            print("Epoch",epoch+1,"-----------------------------------")
            
            for file in trainFiles:
                print("Read:",file)
                
                # Files loop
                for i, data in enumerate(DataLoader(Dataset(file),
                                                    #shuffle     = True,
                                                    pin_memory  = True,
                                                    batch_size  = batch_size,
                                                    num_workers = 4), 0):
                    # get the inputs; data is a list of [frame, steer]
                    frame, steer = data

                    frame = frame.to(device)
                    steer = steer.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    outputs = self.net(frame)
                    loss    = weightedLoss(outputs, steer, weightLoss)
                    loss .backward()
                    optimizer.step()

                    ite = ite + 1

                    # print statistics
                    running_loss += loss.item()
                    if i % stepView == (stepView-1):   # print every stepView mini-batches
                        print(i+1,":\tloss =",running_loss/stepView,"\t\t",timeExecution(ite))
                        running_loss = 0.0
 


"""
Codevilla 2019 Network
----------------------
Ref: 
    https://arxiv.org/pdf/1710.02410.pdf
"""
"""
class CodevillaModel(object):
    def __init__(self):
        # Configure
        self._config = Config
        self.    net = None
        
        # Paths
        trainPath = self._config.trainPath
        validPath = self._config.validPath

        self._trainFile = glob.glob(trainPath + '*.h5')
        self._validFile = glob.glob(validPath + '*.h5')

        # Nets
        self.net = Codevilla19Net(self._config)

    def build(self):
        self.net.build()

    def train(self):
        self.net.fit( self._config.trainPath,
                      self._config.validPath )
""" 
