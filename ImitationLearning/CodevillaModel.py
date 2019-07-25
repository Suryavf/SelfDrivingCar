import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from   torchvision import models, transforms, utils
from   torch.utils.data import Dataset,DataLoader
from   torch.autograd import Variable as V

import ImitationLearning.network.CodevillaNet as imL
from common.data                            import CoRL2017Dataset as Dataset

from config import Config
from config import Global

import common.pytorch as T
from common.utils import averager
from common.utils import iter2time
from common.utils import savePlot
from common.utils import saveHistogram
from common.utils import checkdirectory
from common.utils import cookedFilesList
from common.utils import nameDirectoryModel

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from   tqdm import tqdm
import numpy as np
import cv2 as cv
import secrets
import json
import math
import h5py
import os

# Settings
__global = Global()
__config = Config()

# Conditional (branches)
if __config.model in ['Codevilla18','Codevilla19']:
    __branches = True
else:
    __branches = False

# Multimodal (image + speed)
if __config.model in ['Multimodal','Codevilla18','Codevilla19']:
    __multimodal = True
else:
    __multimodal = False

# Speed regression
if __config.model in ['Codevilla19']:
    __speedReg = True
else:
    __speedReg = False

""" Model prediction
    ----------------
    Predict target by input 
        * Input: model (nn.Module)
                 data  (tuple?)
        * Output: action: Ground truth
                  y_pred: prediction
"""
def pred(model,data):
    action = None 
    y_pred = None

    # Codevilla18, Codevilla19
    if __multimodal and __branches:             
        frame, speed, action, mask = data

        mask   =   mask.to(device)
        frame  =  frame.to(device)
        speed  =  speed.to(device)
        action = action.to(device)

        y_pred = model(frame,speed,mask)
    
    # Multimodal
    elif __multimodal and not __branches:
        frame, speed, action = data

        frame  =  frame.to(device)
        speed  =  speed.to(device)
        action = action.to(device)

        y_pred = model(frame,speed)

    # Basic
    elif not __multimodal and not __branches:
        frame, action = data

        frame  =  frame.to(device)
        action = action.to(device)
        
        y_pred = model(frame)
    else:
        raise NameError('ERROR 404: Model no found')

    return action, y_pred


# Parameters
stepView  = __global.stepView

# CUDA
device = torch.device('cuda:0')

# Settings
_global = Global()
_config = Config()

# Parameters
batch_size     = 120
n_epoch        = 150
framePerSecond =  10

class ImitationModel(object):
    """ Constructor """
    def __init__(self):
        # Paths
        savedPath = _config.savedPath
        modelDir  = savedPath + "/" + nameDirectoryModel(_config.model)
        
        self._figurePath = modelDir  +  "/Figure"
        self. _modelPath = modelDir  +  "/Model"
        self. _validPath = _config.validPath
        self. _trainPath = _config.trainPath
        self.  _cookPath = _config.cookdPath

        checkdirectory(savedPath)
        checkdirectory(modelDir)
        checkdirectory(self._figurePath)
        checkdirectory(self. _modelPath)

        # Nets
        if   _config.model is 'Basic':
            self.net = imL.BasicNet()
        elif _config.model is 'Multimodal':
            self.net = imL.MultimodalNet()
        else:
            print("ERROR: mode no found")
        
        # Save settings
        self.net.saveSettings(modelDir + "/setting.json")

        # Optimizator
        self._optimizer = optim.Adam(   self.net.parameters(), 
                                        lr    = _config.adam_lrate, 
                                        betas =(_config.adam_beta_1, 
                                                _config.adam_beta_2)   )

        # Scheduler
        self._scheduler = optim.lr_scheduler.StepLR(self._optimizer,
                                                    step_size = _config.learning_rate_decay_steps,
                                                    gamma     = _config.learning_rate_decay_factor)

        # Internal parameters
        self._state = {}
        self._trainLoss = list()
        self._validLoss = list()
        self._metrics   = {}

    """ Training state functions """
    def _state_reset(self):
        self._state = {}
    def _state_add(self,name,attr):
        self._state[name]=attr
    def _state_addMetrics(self,metr):
        self._state_add('steerMSE',metr[0,0])
        self._state_add(  'gasMSE',metr[0,1])
        self._state_add('brakeMSE',metr[0,2])
        if _config.model in ['Codevilla19']:
            self._state_add('speedMSE',metr[0,3])
    def _state_save(self,epoch):
        path = self._modelPath + "/model" + str(epoch + 1) + ".pth"
        torch.save(self._state,path)


    """ Save figures """
    def _saveLossFigures(self,epoch,trainLoss,validLoss):
        self._trainLoss.append(trainLoss)
        self._validLoss.append(validLoss)
        if epoch > 4:
            data = list()
            data.append( self._trainLoss )
            data.append( self._validLoss )
            title = ["Train","Validation"]
            path = self._figurePath + "/" + "loss" + str(epoch + 1) + ".png"
            savePlot(data,title,path)
    # https://www.oreilly.com/library/view/python-data-science/9781491912126/ch04.html
    def _saveMetricFigures(self,epoch,metrics):
        if epoch == 0:
            self._metrics["Steer"] = list()
            self._metrics["Gas"  ] = list()
            self._metrics["Brake"] = list()
            if _config.model in ['Codevilla19']:
                self._metrics["Speed"] = list()
        else:
            self._metrics["Steer"].append(metrics[0,0]*1.2)
            self._metrics["Gas"  ].append(metrics[0,1])
            self._metrics["Brake"].append(metrics[0,2])
            if _config.model in ['Codevilla19']:
                self._metrics["Speed"].append(metrics[0,3]*85)
        if epoch > 4:
            epochs = list( range(1,epoch+1) )
            fig  = plt.figure(figsize=(8,5))
            grid = plt.GridSpec(20,20)
            if _config.model in ['Basic','Multimodal','Codevilla18']:
                # Create figure    
                steer = fig.add_subplot(grid[0:8,  : ])
                gas   = fig.add_subplot(grid[ 12:,0:9])
                brake = fig.add_subplot(grid[ 12:,11:])
                # Steer
                steer.plot(epochs,self._metrics["Steer"])
                steer.set_title('Steer')
                steer.set_xlabel('Epoch')
                steer.set_ylabel('Angle')
                steer.set_xlim(1,epoch)
                # Gas
                gas.plot(epochs,self._metrics["Gas"])
                gas.set_title('Gas')
                gas.set_xlabel('Epoch')
                gas.set_xlim(1,epoch)
                # Brake
                brake.plot(epochs,self._metrics["Brake"])
                brake.set_title('Brake')
                brake.set_xlabel('Epoch')
                brake.set_xlim(1,epoch)
            elif _config.model in ['Codevilla19']:
                # Create figure    
                steer = fig.add_subplot(grid[0:8,0:9])
                speed = fig.add_subplot(grid[0:8,11:])
                gas   = fig.add_subplot(grid[12:,0:9])
                brake = fig.add_subplot(grid[12:,11:])
                # Steer
                steer.plot(epochs,self._metrics["Steer"])
                steer.set_title('Steer')
                steer.set_xlabel('Epoch')
                steer.set_ylabel('Angle')
                steer.set_xlim(1,epoch)
                # Speed
                speed.plot(epochs,self._metrics["Speed"])
                speed.set_title('Speed')
                speed.set_xlabel('Epoch')
                speed.set_ylabel('m/s')
                speed.yaxis.set_label_position("right")
                speed.set_xlim(1,epoch)
                # Gas
                gas.plot(epochs,self._metrics["Gas"])
                gas.set_title('Gas')
                gas.set_xlabel('Epoch')
                gas.set_xlim(1,epoch)
                # Brake
                brake.plot(epochs,self._metrics["Brake"])
                brake.set_title('Brake')
                brake.set_xlabel('Epoch')
                brake.set_xlim(1,epoch)
            path = self._figurePath + "/" + "metric" + str(epoch + 1) + ".png"
            plt.savefig(path)
    def _saveHistograms(self,epoch,y_out):
        FigPath    = self._figurePath
        saveHistogram(y_out[:,0], FigPath + "/" + "steer" + str(epoch + 1) + ".png")
        saveHistogram(y_out[:,1], FigPath + "/" +   "gas" + str(epoch + 1) + ".png")
        saveHistogram(y_out[:,2], FigPath + "/" + "brake" + str(epoch + 1) + ".png")
        if _config.model in ['Codevilla19']:
            saveHistogram(y_out[:,3], FigPath + "/" + "speed" + str(epoch + 1) + ".png")


    """ Building """
    def build(self):
        self.net = self.net.float()
        self.net = self.net.apply(T.xavierInit)
        self.net = self.net.to(device)


    """ Train/Evaluation """
    def execute(self):
        # List files
        trainPath = self._trainPath#cookedFilesList(self. _cookPath,'Train')
        validPath = self._validPath#cookedFilesList(self. _cookPath,'Valid')

        optimizer = self._optimizer
        scheduler = self._scheduler
        model     = self.net
        lossFunc  = T.weightedLoss
        
        # Loop over the dataset multiple times
        for epoch in range(n_epoch):
            print("Epoch",epoch+1,"-----------------------------------")
            
            # Train
            #lossTrain = T.train(model,optimizer,scheduler,lossFun,trainPath)
            # Acomulative loss
            running_loss = 0.0
            lossTrain   = averager()

            # Data Loader
            loader = DataLoader( Dataset( trainPath, train      =   True      , 
                                                branches   = __branches  ,
                                                multimodal = __multimodal,
                                                speedReg   = __speedReg ),
                                batch_size  = batch_size,
                                num_workers = 4)
            t = tqdm(iter(loader), leave=False, total=len(loader),desc='Train')#,dynamic_ncols=True)
            # Train
            model.train()
            for i, data in enumerate(t,0):
                scheduler.step()
                
                # Model execute
                action, output = pred(model,data)

                # zero the parameter gradients
                optimizer.zero_grad()

                loss = lossFunc(output, action)
                loss.backward()
                optimizer.step()
                
                # Print statistics
                runtime_loss = loss.item()
                running_loss += runtime_loss
                if i % stepView == (stepView-1):   # print every stepView mini-batches
                    message = 'BatchTrain %i - loss=%.5f'
                    t.set_description( message % ( i+1,running_loss/stepView ))
                    t.refresh()
                    running_loss = 0.0
                lossTrain.update(runtime_loss)

            print("Epoch training loss:",lossTrain.val())


            # Validation
            lossValid,metr,out = T.validation(model,lossFunc,validPath)
            
            # Save checkpoint
            self._state_add(     'epoch',           epoch + 1  )
            self._state_add('state_dict',    model.state_dict())
            self._state_add( 'scheduler',scheduler.state_dict())
            self._state_add( 'optimizer',optimizer.state_dict())
            self._state_add('loss_train',           lossTrain  )
            self._state_add('loss_valid',           lossValid  )
            self._state_addMetrics(metr)
            self._state_save(epoch)

            # Save Figures
            self._saveHistograms (epoch,out)
            self._saveLossFigures(epoch,lossTrain,lossValid)
            
