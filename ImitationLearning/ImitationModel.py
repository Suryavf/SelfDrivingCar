import torch
import torch.optim as optim
import torch.nn.functional as F
from   torch.utils.data import Dataset,DataLoader

import ImitationLearning.network.ImitationNet as imL
import Attention.        network.AttentionNet as attn

from IPython.core.debugger import set_trace

import common.figures as F
import common.  utils as U
from   common.RAdam       import RAdam
from   common.Ranger      import Ranger
from   common.data        import CoRL2017Dataset as Dataset
from   common.prioritized import PrioritizedSamples


from   itertools import zip_longest
from   tqdm import tqdm
import pandas as pd
import numpy  as np
import glob
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

        # Internal parameters
        self.epoch = 0
        
        # Nets
        if   self.setting.model == 'Basic'      : self.model =  imL.      BasicNet()
        elif self.setting.model == 'Multimodal' : self.model =  imL. MultimodalNet()
        elif self.setting.model == 'Codevilla18': self.model =  imL.Codevilla18Net()
        elif self.setting.model == 'Codevilla19': self.model =  imL.Codevilla19Net()
        elif self.setting.model == 'Kim2017'    : self.model = attn.    Kim2017Net()
        else:
            txt = self.setting.model
            print("ERROR: mode no found (" + txt + ")")
        
        if not self.init.is_loadedModel:
            # Paths
            self._checkFoldersToSave()

            # Save settings
            self.  init .save( os.path.join(self._modelPath,   "init.json") )
            self.setting.save( os.path.join(self._modelPath,"setting.json") )
        
        # Objects
        self.optimizer  = None
        self.scheduler  = None
        self.weightLoss = None
        self.lossFunc   = None
        
        # Path files
        self.  trainingFiles = glob.glob(os.path.join(self.setting.general.trainPath,'*.h5'))
        self.validationFiles = glob.glob(os.path.join(self.setting.general.validPath,'*.h5'))

        # Prioritized sampling
        self.framePerFile = self.setting.general.framePerFile
        self.sequence_len = self.setting.general.sequence_len 

        self.slidingWindow           = self.setting.general.slidingWindow
        self.samplesByTrainingFile   = self.framePerFile
        self.samplesByValidationFile = self.framePerFile
        if self.setting.boolean.temporalModel:
            self.samplesByTrainingFile = int( (self.framePerFile - self.sequence_len)/self.slidingWindow + 1 )

        self.samplePriority = PrioritizedSamples( len(self.trainingFiles)*self.samplesByTrainingFile )

        # Datasets
        self.trainDataset = Dataset(self.setting,train= True)
        self.validDataset = Dataset(self.setting,train=False)


    """ Check folders to save """
    def _checkFoldersToSave(self, name = None):
        # Data Path
        self. _validPath = self.setting.general.validPath
        self. _trainPath = self.setting.general.trainPath

        # Root Path
        savedPath = self.setting.general.savedPath
        modelPath = os.path.join(savedPath,self.setting.model)
        if name is not None:
            execPath  = os.path.join(modelPath,  name  )
        else:
            execPath  = os.path.join(modelPath,U.nameDirectory())
        U.checkdirectory(savedPath)
        U.checkdirectory(modelPath)
        U.checkdirectory( execPath)

        # Figures Path
        self._figurePath                = os.path.join(execPath,"Figure")
        self._figurePolarPath           = os.path.join(self._figurePath,"Polar")
        self._figureScatterPath         = os.path.join(self._figurePath,"Scatter")
        self._figureHistogramPath       = os.path.join(self._figurePath,"Histogram")
        self._figureScatterErrorPath    = os.path.join(self._figurePath,"ScatterError")
        self._figureColorMershErrorPath = os.path.join(self._figurePath,"ColorMershError")
        U.checkdirectory(self._figurePath)
        U.checkdirectory(self._figurePolarPath)
        U.checkdirectory(self._figureScatterPath)
        U.checkdirectory(self._figureHistogramPath)
        U.checkdirectory(self._figureScatterErrorPath)
        U.checkdirectory(self._figureColorMershErrorPath)

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
        path = self._modelPath + "/model" + str(epoch) + ".pth"
        torch.save(self._state,path)


    """ Load model """
    def load(self,path):
        # Load
        checkpoint = torch.load(path)
        self.model    .load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint[ 'optimizer'])
        self.scheduler.load_state_dict(checkpoint[ 'scheduler'])

    def to_continue(self,name):
        # Check paths
        self._checkFoldersToSave(name)
        path = U.lastModel(self._modelPath)

        # Next epoch
        self.epoch = int(path.partition('/Model/model')[2].partition('.')[0])
        self.epoch = self.epoch + 1

        # Load
        self.load(path)


    """ Building """
    def build(self):
        self.model = self.model.float()
        self.model = self.model.to(self.device)

        # Optimizator
        if   self.setting.train.optimizer.type ==  "Adam":
            optFun = optim.Adam
        elif self.setting.train.optimizer.type == "RAdam":
            optFun = RAdam
        elif self.setting.train.optimizer.type == "Ranger":
            optFun = Ranger
        else:
            txt = self.setting.train.optimizer.type
            raise NameError('ERROR 404: Optimizer no found ('+txt+')')
        self.optimizer = optFun(    self.model.parameters(),
                                    lr    =  self.setting.train.optimizer.learning_rate, 
                                    betas = (self.setting.train.optimizer.beta_1, 
                                             self.setting.train.optimizer.beta_2 ) )

        # Exploration mode optimizator
        self.exploration_optimizer = optim.SGD( self.model.parameters(), 
                                                lr=self.setting.train.optimizer.learning_rate/100000 )
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.StepLR( self.optimizer,
                                                    step_size = self.setting.train.scheduler.learning_rate_decay_steps,
                                                    gamma     = self.setting.train.scheduler.learning_rate_decay_factor)

        # Loss Function
        self.weightLoss = np.array([self.setting.train.loss.lambda_steer, 
                                    self.setting.train.loss.lambda_gas  , 
                                    self.setting.train.loss.lambda_brake]).reshape(3,1)
        if self.setting.boolean.branches:
            self.weightLoss = np.concatenate( [self.weightLoss for _ in range(4)] )
        self.weightLoss = torch.from_numpy(self.weightLoss).float().cuda(self.device) 

        if self.setting.boolean.speedRegression:
            self.lossFunc = self._weightedLossActSpeed
        else:
            self.lossFunc = self._weightedLossAct


    """ Loss Function """
    def _weightedLossAct(self,measure,prediction,weight=None):
        loss = torch.abs(measure['actions'] - prediction['actions'])
        loss = loss.dot(self.weightLoss)
        if weight is not None:
            # One wight to one sample
            weight = weight.reshape(1,-1)
            ones   = np.ones([self.setting.train.sequence_len,1])
            weight = ones.dot(weight).reshape([-1,1],order='F')
            weight = torch.from_numpy(weight).to(self.device)
            
            loss = loss.mul(weight)
        
        prediction['loss'] = loss
        return torch.mean(loss)

    def _weightedLossActSpeed(self,measure,prediction,weight=None):
        # Action loss
        action = torch.abs(measure['actions'] - prediction['actions'])
        action = action.dot(self.weightLoss)

        # Speed loss
        speed   = torch.abs(measure[ 'speed' ] - prediction[ 'speed' ])
        
        # Total loss
        lambda_action = self.setting.train.loss.lambda_action
        lambda_speed  = self.setting.train.loss.lambda_speed
        loss = lambda_action*action + lambda_speed*speed

        if weight is not None:
            # One wight to one sample
            weight = weight.reshape(1,-1)
            ones   = np.ones([self.setting.train.sequence_len,1])
            weight = ones.dot(weight).reshape([-1,1],order='F')
            weight = torch.from_numpy(weight).to(self.device)
            
            loss = loss.mul(weight)

        prediction['loss'] = loss
        return torch.mean(loss)


    """ Generate ID list """
    def _generateIDlist(self,n_samples,prioritized=False,sequence=False):
        # IDs/weights
        if prioritized:
            IDs = np.array([ np.array(self.samplePriority.sample())  for _ in range(n_samples) ])
            IDs = IDs.T
            IDs = IDs[0]
            weights = IDs[1]
        else:
            IDs = np.array( range(n_samples) )
        # Temporal sliding window
        IDs = IDs*self.slidingWindow
        
        # Sequence
        if sequence:
            sequence_len = self.setting.train.sequence_len
            IDs = [ np.array(range(idx,idx+sequence_len)) for idx in IDs ]
            IDs = np.concatenate(IDs)

        if prioritized: return IDs,weights 
        else:           return IDs
        

    """ Train Routine """
    """
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
            if self.setting.boolean.temporalModel:
                return a_msr.flatten(start_dim=0, end_dim=1), a_pred
            else:
                return a_msr, a_pred

        else:
            a_pred,v_pred = output
            if self.setting.boolean.temporalModel:
                return a_msr.flatten(start_dim=0, end_dim=1), a_pred, v_msr.flatten(start_dim=0, end_dim=1), v_pred
            else:
                return a_msr, a_pred, v_msr, v_pred
    """

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

        if frame.shape[0] != self.setting.train.batch_size:
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
            loss = self.lossFunc( (a_msr,a_pred),1 )
        else:
            a_pred,v_pred = output
            loss = self.lossFunc( (a_msr, a_pred, v_msr, v_pred),1 )

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

    def _stack(self,dataset,IDs):
        group = [ dataset[i] for i in IDs ]
        batch = {}
        for key in group[0]:
            batch[key] = np.stack([data[key] for data in group])
        return batch
    
    def _transfer2device(self,batch):
        inputs    = ['frame','actions','speed','command','mask']
        dev_batch = {}

        for ko in inputs:
            if ko in batch:
                dev_batch[ko] = batch[ko].to(self.device)
        return dev_batch

    def _updatePriority(self,prediction,batchID):
        # Index to update
        IDs = batchID.reshape(-1)
        IDs = [int(IDs[i]/self.slidingWindow) for i in range(0,len(IDs),self.sequence_len)]

        # Losses to update
        losses = prediction['loss']
        losses = losses.view(-1,self.sequence_len).mean(1)
        losses = losses.data.cpu().numpy()

        # Update priority
        for idx,p in zip(IDs,losses):
            self.samplePriority.update(idx,p)


    """ Exploration function
        --------------------
        Global train function
            *  Input: None
            * Output: total_loss (float) 
    """
    def _Exploration(self):
        # Parameters
        n_samples    = len(self.trainingFiles)*self.samplesByTrainingFile
        batch_size   = self.setting.general.batch_size
        sequence_len = self.setting.general.sequence_len
        stepView     = self.setting.general.stepView
        batch_size   = batch_size/sequence_len

        # Loss
        lossExp      = U.averager()
        running_loss = 0

        # ID list
        IDs = self._generateIDlist(n_samples,prioritized=False,sequence=True)

        with torch.no_grad(), tqdm(total=int(n_samples/batch_size)) as pbar:
            for i, batchID in enumerate(zip_longest(*(iter(IDs),) * batch_size)):
                # Batch
                batch     = self._stack(self.trainDataset,batchID)
                dev_batch = self._transfer2device(batch)

                # Model
                dev_pred = self.model(dev_batch)
                dev_loss = self.lossFunc(dev_batch,dev_pred)
                
                # Update priority
                self._updatePriority(dev_pred,batchID)
                
                # Print statistics
                runtime_loss = dev_loss.item()
                running_loss += runtime_loss
                if i % stepView == (stepView-1):   # print every stepView mini-batches
                    message = 'BatchExploration loss=%.7f'
                    pbar.set_description( message % ( running_loss/stepView ))
                    pbar.refresh()
                    running_loss = 0.0
                lossExp.update(runtime_loss)
                pbar.update()
            pbar.close()


    """ Train function
        --------------
        Global train function
            * Input: model     (nn.Module)
                     optimizer (torch.optim)
                     lossFunc  (function)
                     path      (path)
            * Output: total_loss (float) 
    """
    def _Train(self,exploration = False):
        
        # Settings
        running_loss = 0.0
        lossTrain    = U.averager()
        stepView     = self.setting.general.stepView

        # Train configure
        self.model.train()
        self.trainingloader.dataset.train()
        self.trainingloader.dataset.exploration( exploration )

        # Optimizer ( Exploration: SGD 
        #             Train      : Adam )    
        if exploration: optimizer = self.optimizer
        else          : optimizer = self.optimizer
        
        # Train loop
        with tqdm(total=len(self.trainingloader)) as pbar:
            for i, _data in enumerate(self.trainingloader):
                # Model execute
                data, idx= _data

                print("idx:",idx)

                set_trace()
                
                pred   = self._trainRoutine(data)
                weight = self.trainingloader.dataset.weight()
                weight = torch.tensor(weight).to(self.device)
                loss   = self.lossFunc(pred,weight)
                
                # Update priority
                #self.trainingloader.dataset.update(loss.item())
                
                # zero the parameter gradients
                optimizer .zero_grad()
                self.model.zero_grad()

                loss.backward()
                optimizer.step()
                
                # Print statistics
                runtime_loss = loss.item()
                running_loss += runtime_loss
                if i % stepView == (stepView-1):   # print every stepView mini-batches
                    message = 'BatchTrain loss=%.7f'
                    pbar.set_description( message % ( running_loss/stepView ))
                    pbar.refresh()
                    running_loss = 0.0
                lossTrain.update(runtime_loss)
                pbar.update(1)
        
        lossTrain = lossTrain.val()
        print("Epoch training loss:",lossTrain)

        if not exploration:
            self.trainingloader.dataset.saveHistory(os.path.join(self._modelPath,"samples.csv"))

        return lossTrain


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

        # Train configure
        self.model.eval()
        self.validationloader.dataset.eval()
        self.validationloader.dataset.exploration( False )
        
        # Model to evaluation
        with torch.no_grad(), tqdm(total=len(self.validationloader)) as pbar:
            for i, data in enumerate(self.validationloader):
                # Model execute
                loss,err,steer,errSteer,a_pred,v_msr,command= self._validationRoutine(data)
                
                if loss == -1: break
                
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
                    pbar.set_description( message % ( running_loss/stepView ))
                    pbar.refresh()
                    running_loss = 0.0
                pbar.update(1)

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
        colorMershErrorPath = os.path.join(self._figureColorMershErrorPath,"ColorMershError"+str(epoch)+".png")
        scatterErrorPath    = os.path.join(self._figureScatterErrorPath   ,   "ScatterError"+str(epoch)+".png")
        histogramPath       = os.path.join(self._figureHistogramPath      ,      "Histogram"+str(epoch)+".png")
        
        F. saveHistogramSteer(all_action[:,0],histogramPath)
        F.   saveScatterError(all_steer,all_errSteer,all_command,   scatterErrorPath)
        F.saveColorMershError(all_steer,all_errSteer,all_command,colorMershErrorPath)
        
        return running_loss,metrics
    

    """ Train/Evaluation """
    def execute(self):
        # Parameters
        outputSpeed = self.setting.boolean.outputSpeed
        n_epoch     = self.setting.train.n_epoch

        # Plotting objects
        epochLoss  = F.save2PlotByStep(self._figurePath,"Loss","Train","Evaluation")
        epochSteer = F.savePlotByStep (self._figurePath,"Steer")
        epochGas   = F.savePlotByStep (self._figurePath,"Gas"  )
        epochBrake = F.savePlotByStep (self._figurePath,"Brake")
        if outputSpeed:
            epochSpeed = F.savePlotByStep(self._figurePath,"Speed")

        # Initialize
        if self.init.is_loadedModel:
            valuesToSave = U.loadValuesToSave( os.path.join(self._modelPath,"model.csv") )
        else:
            valuesToSave = list()
        df = pd.DataFrame()

        # Loop over the dataset multiple times
        for epoch in range(self.epoch,n_epoch):
            if epoch == 0: print("\nExploration")
            else:          print("\nEpoch",epoch,"-"*40)
            
            # Train
            lossTrain = self._Train( exploration = (epoch==0) )
            if epoch == 0: continue
            else:          self.scheduler.step()
            
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
            df.to_csv(self._modelPath + "/model.csv", index=False)

            # Save checkpoint
            self._state_add(     'epoch',                    epoch  )
            self._state_add('state_dict',self.    model.state_dict())
            self._state_add( 'scheduler',self.scheduler.state_dict())
            self._state_add( 'optimizer',self.optimizer.state_dict())
            self._state_save(epoch)
    

    """ Plot generate"""
    def plot(self,name):
        # Check paths
        self._checkFoldersToSave(name)
        paths = U.modelList(self._modelPath)
        
        # Parameters
        outputSpeed = self.setting.boolean.outputSpeed

        # Plotting objects
        epochSteer = F.savePlotByStep (self._figurePath,"Steer")
        epochGas   = F.savePlotByStep (self._figurePath,"Gas"  )
        epochBrake = F.savePlotByStep (self._figurePath,"Brake")
        if outputSpeed:
            epochSpeed = F.savePlotByStep(self._figurePath,"Speed")

        # Loop paths
        for epoch,path in enumerate(paths,0):
            print("\nEpoch",epoch,"-"*40)

            # Load
            checkpoint = torch.load(path)
            self.model.load_state_dict(checkpoint['state_dict'])
            
            # Validation
            _,metr = self._Validation(epoch)
            
            # Plotting
            epochSteer.update(metr[0])
            epochGas  .update(metr[1])
            epochBrake.update(metr[2])
            if outputSpeed:
                epochSpeed.update(metr[3])
                
