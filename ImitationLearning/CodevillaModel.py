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
from common.utils import saveHistogram
from common.utils import checkdirectory
from common.utils import cookedFilesList
from common.utils import nameDirectoryModel

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
        savedPath = self._config.savedPath
        modelDir  = savedPath + "/" + nameDirectoryModel(__config.model)
        
        self._figurePath = modelDir  +  "/Figure"
        self. _modelPath = modelDir  +  "/Model"
        self.  _cookPath = self._config.cookdPath
        
        checkdirectory(savedPath)
        checkdirectory(modelDir)
        checkdirectory(self._figurePath)
        checkdirectory(self. _modelPath)

        # Nets
        self.net = None

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


    """ Train/Test """
    def execute(self):
        # List files
        trainFiles = cookedFilesList(self. _cookPath,'Train')
        validFiles = cookedFilesList(self. _cookPath,'Valid')
        FigPath    = self._figurePath

        optimizer = self._optimizer
        model     = self.net
        lossFun   = T.weightedLoss
        
        # Loop over the dataset multiple times
        for epoch in range(n_epoch):
            print("Epoch",epoch+1,"-----------------------------------")
            
            # Train
            T.train(model,optimizer,lossFun,trainFiles)
            
            # Validation
            loss,metr,out = T.validation(model,lossFun,validFiles,FigPath)
            
            path = self._modelPath + "model" + str(epoch + 1) + ".pth"
            state = {      'epoch':              epoch + 1,
                      'state_dict':     model.state_dict(),
                       'optimizer': optimizer.state_dict(),
                            'loss':                   loss
                    }
            
            # Save metrics
            if __config.model in ['Basic', 'Multimodal', 'Codevilla18']:
                state['steerMSE'] = metr[0]
                state[  'gasMSE'] = metr[1]
                state['brakeMSE'] = metr[2]
            if __config.model in ['Codevilla19']:
                state['steerMSE'] = metr[0]
                state[  'gasMSE'] = metr[1]
                state['brakeMSE'] = metr[2]
                state['speedMSE'] = metr[3]
            torch.save(state,path)

            # Save Figures
            if __config.model in ['Basic', 'Multimodal', 'Codevilla18']:
                saveHistogram(out[:,0], FigPath + "/" + "steer" + str(epoch + 1) + ".png")
                saveHistogram(out[:,1], FigPath + "/" +   "gas" + str(epoch + 1) + ".png")
                saveHistogram(out[:,2], FigPath + "/" + "brake" + str(epoch + 1) + ".png")

            if __config.model in ['Codevilla19']:
                saveHistogram(out[:,0], FigPath + "/" + "steer" + str(epoch + 1) + ".png")
                saveHistogram(out[:,1], FigPath + "/" +   "gas" + str(epoch + 1) + ".png")
                saveHistogram(out[:,2], FigPath + "/" + "brake" + str(epoch + 1) + ".png")
                saveHistogram(out[:,3], FigPath + "/" + "speed" + str(epoch + 1) + ".png")
            
