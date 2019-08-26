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
import common.figures as F
import common.utils   as U
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

# Parameters
batch_size     = 120
n_epoch        = 150
framePerSecond =  10


class ImitationModel(object):
    """ Constructor """
    def __init__(self,init,setting):

        self.init    =    init
        self.setting = setting

        # Device
        self.device = self.init.device

        # Paths
        self._checkFoldersToSave()

        # Nets
        if   self.setting.model is 'Basic':
            self.model = imL.BasicNet()
        elif self.setting.model is 'Multimodal':
            self.model = imL.MultimodalNet()
        elif self.setting.model is 'Codevilla18': 
            self.model = imL.Codevilla18Net()
        elif self.setting.model is 'Codevilla19':
            self.model = imL.Codevilla19Net()
        elif self.setting.model is 'Kim2017':
            self.model = attn.Kim2017Net()
        else:
            print("ERROR: mode no found")
        
        # Save settings
        self.model.saveSettings(modelDir + "/setting.json")

        # Optimizator
        self.optimizer = optim.Adam(   self.model.parameters(), 
                                        lr    = self.setting.train.optimizer.learning_rate, 
                                        betas =(self.setting.train.optimizer.beta_1, 
                                                self.setting.train.optimizer.beta_2)   )

        # Scheduler
        self.scheduler = optim.lr_scheduler.StepLR( self.optimizer,
                                                    step_size = self.setting.train.scheduler.learning_rate_decay_steps,
                                                    gamma     = self.setting.train.scheduler.learning_rate_decay_factor)

        # Loss Function
        self.weightLoss = torch.Tensor([self.setting.train.loss.lambda_steer, 
                                        self.setting.train.loss.lambda_gas  , 
                                        self.setting.train.loss.lambda_brake]).float().cuda(self.device) 
        if self.setting.boolean.branches:
            self.weightLoss = torch.cat( [self.weightLoss for _ in range(4)] )
        if self.setting.boolean.speedRegression:
            self.lossFunc = self._weightedLossActSpeed
        else:
            self.lossFunc = self._weightedLossAct

        # Internal parameters
        self._state     = {}
        self._trainLoss = list()
        self._validLoss = list()
        self._metrics   = {}


    """ Check folders to save """
    def _checkFoldersToSave(self):
        # Data Path
        self. _validPath = self.setting.general.validPath
        self. _trainPath = self.setting.general.trainPath

        # Root Path
        os.path.join()
        savedPath = self.setting.general.savedPath
        modelPath = os.path.join(savedPath,self.setting.model)
        execPath  = os.path.join(modelPath,U.nameDirectory())
        checkdirectory(savedPath)
        checkdirectory(modelPath)
        checkdirectory( execPath)

        # Figures Path
        self._figurePath          = os.path.join(execPath,"Figure")
        self._figurePolarPath     = os.path.join(self._figurePath,"Polar")
        self._figureScatterPath   = os.path.join(self._figurePath,"Scatter")
        self._figureHistogramPath = os.path.join(self._figurePath,"Histogram")
        checkdirectory(self._figurePath)
        checkdirectory(self._figurePolarPath)
        checkdirectory(self._figureScatterPath)
        checkdirectory(self._figureHistogramPath)

        # Model path
        self._modelPath = os.path.join(execPath,"Model")
        checkdirectory(self._modelPath)


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
        self.model = self.model.to(self.device)


    """ Loss Function """
    def _weightedLossAct(x):
        a_msr,a_pred = x
        loss = torch.abs(a_msr - a_pred)
        loss = torch.mean(loss,0)

    return torch.sum(loss*self.weightLoss)
    def _weightedLossActSpeed(x):
        a_msr, a_pred, v_msr, v_pred = x

        actionLoss = torch.abs (a_msr - a_pred)
        actionLoss = torch.mean(actionLoss,0)
        actionLoss = torch.sum (actionLoss*self.weightLoss)

        speedLoss = torch.abs(v_msr -v_pred)
        speedLoss = torch.mean(speedLoss)

    return   actionLoss * self.setting.train.loss.lambda_action 
           +  speedLoss * self.setting.train.loss.lambda_speed


    """ Train Routine """
    def _trainRoutine(self,data):

        # Boolean conditions
        branches        = self.setting.boolean.branches
        multimodal      = self.setting.boolean.multimodal
        speedRegression = self.setting.boolean.speedRegression
        
        inputSpeed  = self.setting.boolean. inputSpeed
        outputSpeed = self.setting.boolean.outputSpeed

        # Input
        if   not inputSpeed and not branches:
            frame, action = data
            frame =  frame.to(self.device)
            a_msr = action.to(self.device)
            
            output = self.model(frame)

        elif     inputSpeed and not branches:
            frame, speed, action = data

            frame =  frame.to(self.device)
            v_msr =  speed.to(self.device)
            a_msr = action.to(self.device)

            output = self.model(frame,v_msr)

        elif not inputSpeed and     branches:
            frame, action, mask = data

            mask  =   mask.to(self.device)
            frame =  frame.to(self.device)
            a_msr = action.to(self.device)

            output = self.model(frame,mask)

        elif     inputSpeed and     branches:
            frame, speed, action, mask = data

            mask  =   mask.to(self.device)
            frame =  frame.to(self.device)
            v_msr =  speed.to(self.device)
            a_msr = action.to(self.device)
            
            output = self.model(frame,v_msr,mask)

        else:
            raise NameError('ERROR 404: Model no found')

        # Output
        if not outputSpeed:
            a_pred = output
            return a_msr, a_pred

        else:
            a_pred,v_pred = output
            return a_msr, a_pred, v_msr, v_pred


    """ Train function
        --------------
        Global train function
            * Input: model     (nn.Module)
                     optimizer (torch.optim)
                     lossFunc  (function)
                     path      (path)
            * Output: total_loss (float) 
    """
    def _Train(self):
        
        # Acomulative loss
        running_loss = 0.0
        lossTrain   = averager()
        stepView = setting.general.stepView

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
            pred = self._trainRoutine(data)
            loss = self.lossFunc(pred)
            
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
        
        epochLoss  = F.save2PlotByStep(self._figurePath,"Loss","Train","Evaluation")
        epochSteer = F.savePlotByStep (self._figurePath,"Steer")
        epochGas   = F.savePlotByStep (self._figurePath,"Gas")
        epochBrake = F.savePlotByStep (self._figurePath,"Brake")
        
        valuesToSave = list()
        df = pd.DataFrame()

        if _speedReg or _multimodal:
            epochSpeed = F.savePlotByStep(self._figurePath,"Speed")

        # Loop over the dataset multiple times
        for epoch in range(n_epoch):
            print("\nEpoch",epoch+1,"-----------------------------------")
            
            # Train
            lossTrain = self._Train()
            self.scheduler.step()
            
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
            
