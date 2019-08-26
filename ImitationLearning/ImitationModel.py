import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from   torchvision import models, transforms, utils
from   torch.utils.data import Dataset,DataLoader
from   torch.autograd import Variable as V

import ImitationLearning.network.ImitationNet as imL
import Attention.        network.AttentionNet as attn

from config import Config
from config import Global

import common.pytorch as T
import common.figures as fig
from common.utils import averager
from common.utils import iter2time
from common.utils import savePlot
from common.utils import saveHistogram
from common.utils import checkdirectory
from common.utils import cookedFilesList
from common.utils import nameDirectoryModel
from common.data  import CoRL2017Dataset as Dataset

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from   tqdm import tqdm
import pandas as pd
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
    _branches = True
else:
    _branches = False

# Multimodal (image + speed)
if __config.model in ['Multimodal','Codevilla18','Codevilla19']:
    _multimodal = True
else:
    _multimodal = False

# Speed regression
if __config.model in ['Codevilla19']:
    _speedReg = True
else:
    _speedReg = False

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
    if _multimodal and _branches:             
        frame, speed, action, mask = data

        mask   =   mask.to(device)
        frame  =  frame.to(device)
        speed  =  speed.to(device)
        action = action.to(device)

        if _speedReg:
            y_pred,v_pred = model(frame,speed,mask)
            return action, y_pred, speed, v_pred
        else:
            y_pred = model(frame,speed,mask)
    
    # Multimodal
    elif _multimodal and not _branches:
        frame, speed, action = data

        frame  =  frame.to(device)
        speed  =  speed.to(device)
        action = action.to(device)

        y_pred = model(frame,speed)

    # Basic
    elif not _multimodal and not _branches:
        frame, action = data

        frame  =  frame.to(device)
        action = action.to(device)
        
        y_pred = model(frame)
    else:
        raise NameError('ERROR 404: Model no found')

    return action, y_pred



class ImitationModel(object):
    """ Constructor """
    def __init__(self,init,setting):

        self.init    =    init
        self.setting = setting


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

        checkdirectory(self._figurePath + "/Histogram")
        checkdirectory(self._figurePath + "/Scatter")
        checkdirectory(self._figurePath + "/Polar")

        # Nets
        if   _config.model is 'Basic':
            self.model = imL.BasicNet()
        elif _config.model is 'Multimodal':
            self.model = imL.MultimodalNet()
        elif _config.model is 'Codevilla18':
            self.model = imL.Codevilla18Net()
        elif _config.model is 'Codevilla19':
            self.model = imL.Codevilla19Net()
        elif _config.model is 'Kim2017':
            self.model = attn.Kim2017Net()
        else:
            print("ERROR: mode no found")
        
        # Save settings
        self.model.saveSettings(modelDir + "/setting.json")

        # Optimizator
        self.optimizer = optim.Adam(   self.model.parameters(), 
                                        lr    = _config.adam_lrate, 
                                        betas =(_config.adam_beta_1, 
                                                _config.adam_beta_2)   )

        # Scheduler
        self.scheduler = optim.lr_scheduler.StepLR( self.optimizer,
                                                    step_size = _config.learning_rate_decay_steps,
                                                    gamma     = _config.learning_rate_decay_factor)

        # Loss Function
        self.lossFunc = T.weightedLoss

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
        self._state_add('steerMSE',metr[0])
        self._state_add(  'gasMSE',metr[1])
        self._state_add('brakeMSE',metr[2])
        if _config.model in ['Codevilla19']:
            self._state_add('speedMSE',metr[3])
    def _state_save(self,epoch):
        path = self._modelPath + "/model" + str(epoch + 1) + ".pth"
        torch.save(self._state,path)


    """ Building """
    def build(self):
        self.model = self.model.float()
        self.model = self.model.apply(T.xavierInit)
        self.model = self.model.to(device)


    def trainExecute(self,data):
        # Boolean conditions
        branches        = self.setting.boolean.branches
        multimodal      = self.setting.boolean.multimodal
        speedRegression = self.setting.boolean.speedRegression
        
        inputSpeed = multimodal or speedRegression

        # Input
        if   not inputSpeed and not branches:
            frame, action = data
            
            frame  =  frame.to(device)
            action = action.to(device)

        elif     inputSpeed and not branches:
            frame, speed, action = data

            frame  =  frame.to(device)
            speed  =  speed.to(device)
            action = action.to(device)

        elif not inputSpeed and     branches:
            frame, action, mask = data

            mask   =   mask.to(device)
            frame  =  frame.to(device)
            action = action.to(device)

        elif     inputSpeed and     branches:
            frame, speed, action, mask = data

            mask   =   mask.to(device)
            frame  =  frame.to(device)
            speed  =  speed.to(device)
            action = action.to(device)

        else:
            raise NameError('ERROR 404: Model no found')


        action = None 
        y_pred = None

        # Codevilla18, Codevilla19
        if _multimodal and _branches:             
            frame, speed, action, mask = data

            mask   =   mask.to(device)
            frame  =  frame.to(device)
            speed  =  speed.to(device)
            action = action.to(device)

            if _speedReg:
                y_pred,v_pred = self.model(frame,speed,mask)
                return action, y_pred, speed, v_pred
            else:
                y_pred = self.model(frame,speed,mask)
        
        # Multimodal
        elif _multimodal and not _branches:
            frame, speed, action = data

            frame  =  frame.to(device)
            speed  =  speed.to(device)
            action = action.to(device)

            y_pred = self.model(frame,speed)

        # Basic
        elif not _multimodal and not _branches:
            frame, action = data

            frame  =  frame.to(device)
            action = action.to(device)
            
            y_pred = self.model(frame)
        else:
            raise NameError('ERROR 404: Model no found')

        return action, y_pred


    """ Train function
        --------------
        Global train function
            * Input: model     (nn.Module)
                     optimizer (torch.optim)
                     lossFunc  (function)
                     path      (path)
            * Output: total_loss (float) 
    """
    def train(self):
        
        # Acomulative loss
        running_loss = 0.0
        lossTrain   = averager()

        # Data Loader
        loader = DataLoader(  Dataset(  self.setting.general.trainPath, 
                                        train       = True     , 
                                        branches    = self.setting.boolean.branches  ,
                                        multimodal  = self.setting.boolean.multimodal,
                                        speedReg    = self.setting.boolean.speedRegression ),
                                        batch_size  = self.setting.train.batch_size,
                                        num_workers = self.init.num_workers)
        t = tqdm(iter(loader), leave=False, total=len(loader))
        
        # Train
        self.model.train()
        for i, data in enumerate(t,0):
            # Model execute
            pred = self.trainExecute(data)
            loss = self.lossFunc(1,2)
            """
            if _speedReg:
                action, y_pred, speed, v_pred = pred(model,data)
                loss = T.weightedSpeedRegLoss(y_pred,action,speed,v_pred)
            else:
                action, output = pred(model,data)
                loss = T.weightedLoss(output, action)
            """

            # zero the parameter gradients
            self.optimizer.zero_grad()
            self.model    .zero_grad()

            loss.backward()
            self.optimizer.step()
            
            # Print statistics
            runtime_loss = loss.item()
            running_loss += runtime_loss
            if i % stepView == (stepView-1):   # print every stepView mini-batches
                message = 'BatchTrain loss=%.7f'
                t.set_description( message % ( running_loss/stepView ))
                t.refresh()
                running_loss = 0.0
            lossTrain.update(runtime_loss)
        t.close()
        
        lossTrain = lossTrain.val()
        print("Epoch training loss:",lossTrain)

        return lossTrain





    """ Train/Evaluation """
    def execute(self):
        # List files
        trainPath = self._trainPath
        validPath = self._validPath

        optimizer = self._optimizer
        scheduler = self._scheduler
        model     = self.model
        lossFunc  = T.weightedLoss
        
        epochLoss  = fig.save2PlotByStep(self._figurePath,"Loss","Train","Evaluation")
        epochSteer = fig.savePlotByStep (self._figurePath,"Steer")
        epochGas   = fig.savePlotByStep (self._figurePath,"Gas")
        epochBrake = fig.savePlotByStep (self._figurePath,"Brake")
        
        valuesToSave = list()
        df = pd.DataFrame()

        if _speedReg or _multimodal:
            epochSpeed = fig.savePlotByStep(self._figurePath,"Speed")

        # Loop over the dataset multiple times
        for epoch in range(n_epoch):
            print("\nEpoch",epoch+1,"-----------------------------------")
            
            # Train
            #lossTrain = T.train(model,optimizer,scheduler,lossFun,trainPath)
            # Acomulative loss
            running_loss = 0.0
            lossTrain   = averager()
            
            # Data Loader
            loader = DataLoader( Dataset( trainPath, train       =   True     , 
                                                     branches    = _branches  ,
                                                     multimodal  = _multimodal,
                                                     speedReg    = _speedReg ),
                                                     batch_size  =  batch_size,
                                                     num_workers =  8)
            t = tqdm(iter(loader), leave=False, total=len(loader))
        
            # Train
            model.train()
            for i, data in enumerate(t,0):
                # Model execute
                if _speedReg:
                    action, y_pred, speed, v_pred = pred(model,data)
                    loss = T.weightedSpeedRegLoss(y_pred,action,speed,v_pred)
                else:
                    action, output = pred(model,data)
                    loss = T.weightedLoss(output, action)

                # zero the parameter gradients
                optimizer.zero_grad()
                model    .zero_grad()

                loss.backward()
                optimizer.step()
                
                # Print statistics
                runtime_loss = loss.item()
                running_loss += runtime_loss
                if i % stepView == (stepView-1):   # print every stepView mini-batches
                    message = 'BatchTrain loss=%.7f'
                    t.set_description( message % ( running_loss/stepView ))
                    t.refresh()
                    running_loss = 0.0
                lossTrain.update(runtime_loss)
            t.close()
            
            # Scheduler step
            scheduler.step()
            
            lossTrain = lossTrain.val()
            print("Epoch training loss:",lossTrain)

            # Validation
            lossValid,metr = T.validation(model,lossFunc,epoch,validPath,self._figurePath)
            
            epochLoss. update(lossTrain,lossValid)
            epochSteer.update(metr[0])
            epochGas  .update(metr[1])
            epochBrake.update(metr[2])
            if _speedReg:
                epochSpeed.update(metr[3])

            if _speedReg:
                valuesToSave.append( (lossTrain,lossValid,metr[0],metr[1],metr[2],metr[3]) )
                df = pd.DataFrame(valuesToSave, columns = ['LossTrain','LossValid','Steer','Gas','Brake','Speed'])
            else:
                valuesToSave.append( (lossTrain,lossValid,metr[0],metr[1],metr[2]) )
                df = pd.DataFrame(valuesToSave, columns = ['LossTrain','LossValid','Steer','Gas','Brake'])

            # Save csv
            df.to_csv(self._modelPath + "/model.csv")

            # Save checkpoint
            self._state_add(     'epoch',           epoch + 1  )
            self._state_add('state_dict',    model.state_dict())
            self._state_add( 'scheduler',scheduler.state_dict())
            self._state_add( 'optimizer',optimizer.state_dict())
            self._state_add('loss_train',           lossTrain  )
            self._state_add('loss_valid',           lossValid  )
            self._state_addMetrics(metr)
            self._state_save(epoch)
            
