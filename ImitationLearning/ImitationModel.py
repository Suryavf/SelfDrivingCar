import torch
import torch.optim as optim
import torch.nn.functional as F
from   torch.utils.data import Dataset,DataLoader

import ImitationLearning.network.ImitationNet as imL
import Attention.        network.AttentionNet as attn

import common.figures as F
import common.  utils as U
from common.RAdam  import RAdam
from common.Ranger import Ranger
from common.data  import CoRL2017Dataset as Dataset

from   tqdm import tqdm
import pandas as pd
import numpy  as np
import os

# Solution DataLoader bug
# Ref: https://github.com/pytorch/pytorch/issues/973
# import resource
# rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
# resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))
# =================
torch.multiprocessing.set_sharing_strategy('file_system')
# =================

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
        self.  init .save( os.path.join(self._modelPath,   "init.json") )
        self.setting.save( os.path.join(self._modelPath,"setting.json") )

        # Optimizator
        if   self.setting.train.optimizer.type is "Adam":
            optFun = optim.Adam
        elif self.setting.train.optimizer.type is "RAdam":
            optFun = RAdam
        elif self.setting.train.optimizer.type is "Ranger":
            optFun = Ranger
        else:
            raise NameError('ERROR 404: Optimizer no found')
        self.optimizer = optFun(    self.model.parameters(),
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
        savedPath = self.setting.general.savedPath
        modelPath = os.path.join(savedPath,self.setting.model)
        execPath  = os.path.join(modelPath,U.nameDirectory())
        U.checkdirectory(savedPath)
        U.checkdirectory(modelPath)
        U.checkdirectory( execPath)

        # Figures Path
        self._figurePath             = os.path.join(execPath,"Figure")
        self._figurePolarPath        = os.path.join(self._figurePath,"Polar")
        self._figureScatterPath      = os.path.join(self._figurePath,"Scatter")
        self._figureHistogramPath    = os.path.join(self._figurePath,"Histogram")
        self._figureScatterErrorPath = os.path.join(self._figurePath,"ScatterError")
        U.checkdirectory(self._figurePath)
        U.checkdirectory(self._figurePolarPath)
        U.checkdirectory(self._figureScatterPath)
        U.checkdirectory(self._figureHistogramPath)
        U.checkdirectory(self._figureScatterErrorPath)

        # Model path
        self._modelPath = os.path.join(execPath,"Model")
        U.checkdirectory(self._modelPath)


    """ Training state functions """
    def _state_reset(self):
        self._state = {}
    def _state_add(self,name,attr):
        self._state[name]=attr
    def _state_addMetrics(self,metr):
        self._state_add('steerMSE',metr[0])
        self._state_add(  'gasMSE',metr[1])
        self._state_add('brakeMSE',metr[2])
        if self.setting.boolean.speedRegression:
            self._state_add('speedMSE',metr[3])
    def _state_save(self,epoch):
        path = self._modelPath + "/model" + str(epoch + 1) + ".pth"
        torch.save(self._state,path)


    """ Load model """
    def load(self,path):
        checkpoint = torch.load(path)
        
        self.model    .load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint[ 'optimizer'])
        self.scheduler.load_state_dict(checkpoint[ 'scheduler'])


    """ Building """
    def build(self):
        self.model = self.model.float()
        self.model = self.model.to(self.device)


    """ Loss Function """
    def _weightedLossAct(self,x):
        a_msr,a_pred = x
        loss = torch.abs(a_msr - a_pred)
        loss = torch.mean(loss,0)

        return torch.sum(loss*self.weightLoss)
    def _weightedLossActSpeed(self,x):
        a_msr, a_pred, v_msr, v_pred = x

        actionLoss = torch.abs (a_msr - a_pred)
        actionLoss = torch.mean(actionLoss,0)
        actionLoss = torch.sum (actionLoss*self.weightLoss)

        speedLoss = torch.abs(v_msr -v_pred)
        speedLoss = torch.mean(speedLoss)

        lambda_action = self.setting.train.loss.lambda_action
        lambda_speed  = self.setting.train.loss.lambda_speed
        return actionLoss * lambda_action  +  speedLoss * lambda_speed


    """ Train Routine """
    def _trainRoutine(self,data):

        # Boolean conditions
        branches    = self.setting.boolean.   branches
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
        lossTrain   = U.averager()
        stepView = self.setting.general.stepView

        # Data Loader
        loader = DataLoader(Dataset(self.setting, train = True),
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


    """ Validation Routine """
    def _validationRoutine(self,data):
        # Boolean conditions
        branches    = self.setting.boolean.  branches
        inputSpeed  = self.setting.boolean.inputSpeed

        # Input
        if self.setting.boolean.branches:
            frame, command, v_msr, a_msr, mask = data
            mask    =    mask.to(self.device)
            frame   =   frame.to(self.device)
            v_msr   =   v_msr.to(self.device)
            a_msr   =   a_msr.to(self.device)
            command = command.to(self.device)

        else:
            frame, command, v_msr, a_msr = data
            frame   =   frame.to(self.device)
            v_msr   =   v_msr.to(self.device)
            a_msr   =   a_msr.to(self.device)
            command = command.to(self.device)

        if frame.shape[0] != 120:
            loss,err,steer,errSteer,a_pred,v_msr,command = (-1,-1,-1,-1,-1,-1,-1)
            return loss,err,steer,errSteer,a_pred,v_msr,command
        
        # Model
        if   not inputSpeed and not branches:
            output = self.model(frame)
        elif     inputSpeed and not branches:
            output = self.model(frame,v_msr)
        elif not inputSpeed and     branches:
            output = self.model(frame,mask)
        elif     inputSpeed and     branches:
            output = self.model(frame,v_msr,mask)
        else:
            raise NameError('ERROR 404: Model no found')

        # Loss
        if not self.setting.boolean.outputSpeed:
            a_pred = output
            loss = self.lossFunc( (a_msr,a_pred) )
        else:
            a_pred,v_pred = output
            loss = self.lossFunc( (a_msr, a_pred, v_msr, v_pred) )

        # Error
        err = torch.abs(a_msr - a_pred)
        if self.setting.boolean.branches:
            err = err.view(-1,4,3)
            err = err.sum(1)
        errSteer = err[:,0]
        err = torch.mean(err,0)
        if self.setting.boolean.outputSpeed:
            error = torch.zeros(4)
            error[:3] = err
            verr = torch.abs(v_msr - v_pred)
            verr = torch.mean(verr)
            error[3] = verr
            err = error

        # Action
        if self.setting.boolean.branches:
            a_pred = a_pred.view(-1,4,3)
            a_pred = a_pred.sum(1)

        # Steering
        if self.setting.boolean.branches:
            a_msr = a_msr.view(-1,4,3)
            a_msr = a_msr.sum(1)
        steer = a_msr[:,0]

        return loss,err,steer,errSteer,a_pred,v_msr,command


    """ Validation function
        -------------------
        Global validation function
            * Input: model    (nn.Module)
                     lossFunc (function)
                     path     (path)
            * Output: total_loss (float) 
    """
    def _Validation(self,epoch):
        # Acomulative loss
        running_loss = 0.0
        all_speed    = list()
        all_steer    = list()
        all_action   = list()
        all_command  = list()
        all_errSteer = list()
        lossValid = U.averager()
        stepView  = self.setting.general.stepView
        
        max_speed    = self.setting.preprocessing.max_speed
        max_steering = self.setting.preprocessing.max_steering

        # Metrics
        if self.setting.boolean.speedRegression:
            metrics = U.averager(4)   # Steer,Gas,Brake,Speed
        else:
            metrics = U.averager(3)   # Steer,Gas,Brake

        # Data Loader
        loader = DataLoader(Dataset(self.setting, train = False),
                                    batch_size  = self.setting.train.batch_size,
                                    num_workers = self.init.num_workers)
        t = tqdm(iter(loader), leave=False, total=len(loader))
        
        # Model to evaluation
        self.model.eval()
        with torch.no_grad():
            for i, data in enumerate(t,0):
                # Model execute
                loss,err,steer,errSteer,a_pred,v_msr,command= self._validationRoutine(data)
                
                if loss == -1:
                    break
                
                # Calculate the loss
                runtime_loss  = loss.item()
                running_loss += runtime_loss
                lossValid.update(runtime_loss)
                
                # Metrics
                metrics.update(err.data.cpu().numpy())

                # Save values
                all_speed   .append(    v_msr.data.cpu().numpy() )
                all_steer   .append(    steer.data.cpu().numpy() )
                all_action  .append(   a_pred.data.cpu().numpy() )
                all_command .append(  command.data.cpu().numpy() )
                all_errSteer.append( errSteer.data.cpu().numpy() )

                # Print statistics
                if i % stepView == (stepView-1):   # print every stepView mini-batches
                    message = 'BatchValid loss=%.7f'
                    t.set_description( message % ( running_loss/stepView ))
                    t.refresh()
                    running_loss = 0.0
                t.update()
            t.close()

        # Loss/metrics
        metrics      =    metrics.mean
        running_loss = lossValid.val()
        
        # Concatenate List
        all_errSteer = np.concatenate(all_errSteer,0)
        all_command  = np.concatenate(all_command ,0)
        all_action   = np.concatenate(all_action  ,0)
        all_steer    = np.concatenate(all_steer   ,0)
        all_speed    = np.concatenate(all_speed   ,0)

        # To real values
        metrics   [  0] = metrics   [  0] * max_steering
        all_action[:,0] = all_action[:,0] * max_steering
        all_errSteer    = all_errSteer    * max_steering
        all_steer       = all_steer       * max_steering
        all_speed       = all_speed       * max_speed
        if self.setting.boolean.speedRegression:
            metrics [3] = metrics[3]      * max_speed

        # Print results
        print("Validation loss:",running_loss)
        if self.setting.boolean.speedRegression:
            print("Steer:",metrics[0],"\tGas:",metrics[1],"\tBrake:",metrics[2],"\tSpeed:",metrics[3])
        else:
            print("Steer:",metrics[0],"\tGas:",metrics[1],"\tBrake:",metrics[2])
        
        # Save figures
        scatterErrorPath = os.path.join(self._figureScatterErrorPath,"ScatterError"+str(epoch+1)+".png")
        histogramPath    = os.path.join(self._figureHistogramPath   ,   "Histogram"+str(epoch+1)+".png")
        scatterPath      = os.path.join(self._figureScatterPath     ,     "Scatter"+str(epoch+1)+".png")
        polarPath        = os.path.join(self._figurePolarPath       ,       "Polar"+str(epoch+1)+".png")
        F.saveHistogramSteer        (all_action[:,0],histogramPath)
        F.saveScatterSteerSpeed     (all_action[:,0],all_speed,all_command, scatterPath )
        F.saveScatterPolarSteerSpeed(all_action[:,0],all_speed,all_command,   polarPath )
        
        F.saveScatterError(all_steer,all_errSteer,all_command,scatterErrorPath)

        return running_loss,metrics
    

    """ Train/Evaluation """
    def execute(self):
        # Boolean conditions
        outputSpeed = self.setting.boolean.outputSpeed

        epochLoss  = F.save2PlotByStep(self._figurePath,"Loss","Train","Evaluation")
        epochSteer = F.savePlotByStep (self._figurePath,"Steer")
        epochGas   = F.savePlotByStep (self._figurePath,"Gas")
        epochBrake = F.savePlotByStep (self._figurePath,"Brake")
        
        valuesToSave = list()
        n_epoch = self.setting.train.n_epoch
        df = pd.DataFrame()

        if outputSpeed:
            epochSpeed = F.savePlotByStep(self._figurePath,"Speed")

        # Loop over the dataset multiple times
        for epoch in range(n_epoch):
            print("\nEpoch",epoch+1,"-----------------------------------")
            
            # Train
            lossTrain = self._Train()
            self.scheduler.step()
            
            # Validation
            lossValid,metr = self._Validation(epoch)
            
            epochLoss. update(lossTrain,lossValid)
            epochSteer.update(metr[0])
            epochGas  .update(metr[1])
            epochBrake.update(metr[2])
            if outputSpeed:
                epochSpeed.update(metr[3])

            if outputSpeed:
                valuesToSave.append( (lossTrain,lossValid,metr[0],metr[1],metr[2],metr[3]) )
                df = pd.DataFrame(valuesToSave, columns = ['LossTrain','LossValid','Steer','Gas','Brake','Speed'])
            else:
                valuesToSave.append( (lossTrain,lossValid,metr[0],metr[1],metr[2]) )
                df = pd.DataFrame(valuesToSave, columns = ['LossTrain','LossValid','Steer','Gas','Brake'])

            # Save csv
            df.to_csv(self._modelPath + "/model.csv")

            # Save checkpoint
            self._state_add(     'epoch',                epoch + 1  )
            self._state_add('state_dict',self.    model.state_dict())
            self._state_add( 'scheduler',self.scheduler.state_dict())
            self._state_add( 'optimizer',self.optimizer.state_dict())
            self._state_add('loss_train',                lossTrain  )
            self._state_add('loss_valid',                lossValid  )
            self._state_addMetrics(metr)
            self._state_save(epoch)
            
