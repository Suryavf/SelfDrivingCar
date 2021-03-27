import os
import glob

import h5py
import pickle
from   tqdm import tqdm
import pandas as pd
import numpy  as np

import torch
import torch.optim as optim
import torch.nn    as    nn
from   torch.utils.data        import DataLoader
from   torch.utils.tensorboard import SummaryWriter

import StateOfArt.ImitationLearning.ImitationNet as imL
import StateOfArt.        Attention.AttentionNet as attn
import ImitationLearning.VisualAttention.Model   as exper

from IPython.core.debugger import set_trace

import ImitationLearning.VisualAttention.Decoder           as D
import ImitationLearning.VisualAttention.Encoder           as E
import ImitationLearning.VisualAttention.network.Attention as A
import ImitationLearning.VisualAttention.network.Control   as C

import common.directory as V
import common.  figures as F
import common.    utils as U
import common.     loss as L
from   common.RAdam       import RAdam
from   common.Ranger      import Ranger
from   common.DiffGrad    import DiffGrad
from   common.DiffRGrad   import DiffRGrad
from   common.DeepMemory  import DeepMemory
from   common.data        import CoRL2017Dataset
from   common.data        import CARLA100Dataset
from   common.data        import  GeneralDataset as Dataset
from   common.prioritized import PrioritizedSamples

# Solution DataLoader bug
# Ref: https://github.com/pytorch/pytorch/issues/973
# import resource
# rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
# resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))
# =================
torch.multiprocessing.set_sharing_strategy('file_system')
# =================

# GPU trick
# https://nvlabs.github.io/eccv2020-mixed-precision-tutorial/files/szymon_migacz-pytorch-performance-tuning-guide.pdf
# For convolutional neural networks, enable cuDNN autotuner by setting:
torch.backends.cudnn.benchmark = True

class ImitationModel(object):
    """ Constructor """
    def __init__(self,init,setting):

        self.init    =    init
        self.setting = setting

        # Device
        self.device = self.init.device

        # Internal parameters
        self.codename  = None
        self.samplesID = {}
        self._state    = {}
        self. epoch    = 0
        
        # Development settings
        self.save_priority_history  = False
        self.save_speed_action_plot = False
        self.speed_regularization   = self.setting.train.loss.type in ["WeightedReg","WeightedMultiTask"]
        
        # Objects
        self.optimizer = None
        self.scheduler = None
        self.lossFunc  = None
        self.lossEval  = None
        self.model     = None
        
        # Dataset
        self.trainDataset = None
        self.validDataset = None
        self.n_training   = None
        self.n_validation = None

        self.framePerFile  = self.setting.general. framePerFile
        self.sequence_len  = self.setting.general. sequence_len 
        self.slidingWindow = self.setting.general.slidingWindow

        self.dataset  = self.setting.general.dataset
        self.CoRL2017 = (self.dataset == 'CoRL2017')

        # Paths
        self. _codePath  = None
        self._modelPath  = None
        self._trainPath  = None
        self._validPath  = None
        self._figurePath = None
        self._figureSteerErrorPath = None
        self._figureGasErrorPath   = None
        self._figureBrakeErrorPath = None

        # Prioritized sampling
        temp = self.setting.boolean.temporalModel
        if temp: self.samplesByTrainingFile = int( (self.framePerFile - self.sequence_len)/self.slidingWindow + 1 )
        else   : self.samplesByTrainingFile = self.framePerFile
        self.samplesByValidationFile = self.framePerFile
        self.samplePriority = None # object
        

    """ Check folders to save """
    def _checkFoldersToSave(self, name = None):
        # Data Path
        self._validPath = self.setting.general.validPath
        self._trainPath = self.setting.general.trainPath

        # Root Path
        savedPath = self.setting.general.savedPath
        modelPath = os.path.join(savedPath,self.setting.model)
        if name is not None: self.codename = name
        else               : self.codename = U.nameDirectory()
        execPath = os.path.join(modelPath,self.codename )
        U.checkdirectory(savedPath)
        U.checkdirectory(modelPath)
        U.checkdirectory( execPath)
        print("Execute %s model: %s\n"%(self.setting.model,self.codename))

        # Figures Path
        self._figurePath           = os.path.join(execPath,"Figure")
        self._figureSteerErrorPath = os.path.join(self._figurePath,"SteerError")
        
        U.checkdirectory(self._figurePath)
        U.checkdirectory(self._figureSteerErrorPath)

        if self.save_speed_action_plot:
            self._figureGasErrorPath   = os.path.join(self._figurePath,  "GasError")
            self._figureBrakeErrorPath = os.path.join(self._figurePath,"BrakeError")
        
            U.checkdirectory(self._figureGasErrorPath  )
            U.checkdirectory(self._figureBrakeErrorPath)

        # Model path
        self._codePath  = execPath
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
        if self.setting.boolean.outputSpeed:
            self._state_add('speedMSE',metr[3])
    def _state_save(self,epoch):
        # Save model
        pathMod = os.path.join( self._modelPath, "model" + str(epoch) + ".pth" )
        torch.save( self._state, pathMod)

        # Priority save
        if self.save_priority_history:
            pathPri = os.path.join( self._modelPath, "priority" + str(epoch) + ".pck" )
        else: 
            pathPri = os.path.join( self._modelPath, "priority.pck" )
        self.samplePriority.save(pathPri)


    """ Load model """
    def load(self,path):
        # Load
        checkpoint = torch.load(path)
        self.model    .load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint[ 'optimizer'])
        self.scheduler.load_state_dict(checkpoint[ 'scheduler'])
        self.samplePriority.load(os.path.join(self._modelPath,"priority.pck"))
        
    def create_model(self):
        # Paths
        self._checkFoldersToSave()

        # Save settings
        self.  init .save( os.path.join(self._modelPath,   "init.json") )
        self.setting.save( os.path.join(self._modelPath,"setting.json") )

    def to_continue(self,name,epoch = None, study=False):
        # Check paths
        self._checkFoldersToSave(name)
        path = U.lastModel(self._modelPath)

        # Epoch value
        if epoch is None: self.epoch = int(path.partition('/Model/model')[2].partition('.')[0])
        else            : self.epoch = epoch

        # Load
        if not study: self.load(path)


    """ Building """
    def build(self,study=False):
        
        # Model
        if   self.setting.model == 'Basic'       : self.model =  imL.      BasicNet()
        elif self.setting.model == 'Multimodal'  : self.model =  imL. MultimodalNet()
        elif self.setting.model == 'Codevilla18' : self.model =  imL.Codevilla18Net()
        elif self.setting.model == 'Codevilla19' : self.model =  imL.Codevilla19Net()
        elif self.setting.model == 'Kim2017'     : self.model = attn.    Kim2017Net()
        elif self.setting.model == 'Experimental': self.model = exper. Experimental(self.setting)
        elif self.setting.model == 'ExpBranch'   : self.model = exper.    ExpBranch(self.setting)
        elif self.setting.model == 'Approach'    : self.model = exper.     Approach(self.setting,study)
        else:
            txt = self.setting.model
            print("ERROR: mode no found (" + txt + ")")
        self.model = self.model.float()
        self.model = self.model.to(self.device)

        # Optimizator
        if   self.setting.train.optimizer.type == "Adam":
            optFun = optim.Adam
        elif self.setting.train.optimizer.type == "RAdam":
            optFun = RAdam
        elif self.setting.train.optimizer.type == "Ranger":
            optFun = Ranger
        elif self.setting.train.optimizer.type == "DiffGrad":
            optFun = DiffGrad
        elif self.setting.train.optimizer.type == "DiffRGrad":
            optFun = DiffRGrad
        elif self.setting.train.optimizer.type == "DeepMemory":
            optFun = DeepMemory
        else:
            txt = self.setting.train.optimizer.type
            raise NameError('ERROR 404: Optimizer no found ('+txt+')')
        self.optimizer = optFun(    self.model.parameters(),
                                    lr    =  self.setting.train.optimizer.learningRate, 
                                    betas = (self.setting.train.optimizer.beta1, 
                                             self.setting.train.optimizer.beta2 ) )

        # Scheduler
        self.scheduler = optim.lr_scheduler.StepLR( self.optimizer,
                                                    step_size = self.setting.train.scheduler.learning_rate_decay_steps,
                                                    gamma     = self.setting.train.scheduler.learning_rate_decay_factor)

        # Loss Function
        if   self.setting.train.loss.type == "Weighted":
            self.lossFunc = L.WeightedLoss(self.init,self.setting)
            self. lossEval = self.lossFunc.eval
        elif self.setting.train.loss.type == "WeightedReg":
            self.lossFunc = L.WeightedLossReg(self.init,self.setting)
            self. lossEval = self.lossFunc.eval
        elif self.setting.train.loss.type == "WeightedMultiTask":
            self.lossFunc = L.MultitaskLoss(self.init,self.setting)
            self. lossEval = self.lossFunc.eval
        else:
            txt = self.setting.train.loss.type
            raise NameError('ERROR 404: Loss no found ('+txt+')')

        # Build dataset
        samplesPerFile = int( (self.framePerFile - self.sequence_len)/self.slidingWindow + 1 ) 
        if self.CoRL2017:
            # Training data
            if not study:
                trainingFiles = glob.glob(os.path.join(self.setting.general.trainPath,'*.h5'))
                trainingFiles.sort()
                self.trainDataset = CoRL2017Dataset(self.setting,trainingFiles,train= True)
                self.n_training = len(trainingFiles)*samplesPerFile

            # Validation data
            validationFiles = glob.glob(os.path.join(self.setting.general.validPath,'*.h5'))
            validationFiles.sort()
            self.validDataset = CoRL2017Dataset(self.setting,validationFiles,train=False)
            self.n_validation = len(validationFiles)*self.framePerFile

        else:
            # Training data
            if not study:
                self.trainDataset = CARLA100Dataset(self.setting,train=True)
                self.n_training = int(len(self.trainDataset)/self.sequence_len)

            # Validation data
            self.validDataset = CARLA100Dataset(self.setting,train=False)
            self.n_validation = int(len(self.validDataset)/self.sequence_len)

        # Prioritized sampling
        if not study:
            self.samplePriority = PrioritizedSamples( n_samples = self.n_training, 
                                                      alpha = self.setting.sampling.alpha,
                                                      beta  = self.setting.sampling. beta,
                                                      betaLinear = self.setting.sampling.betaLinear,
                                                      betaPhase  = self.setting.sampling.betaPhase,
                                                      balance = self.setting.sampling.balance,
                                                      c    = self.setting.sampling.c,
                                                      fill = not self.CoRL2017)
        

    """ Generate ID list """
    def _samplingPrioritizedSamples(self,n_samples):
        # IDs/weights
        val = np.array([ np.array(self.samplePriority.sample()) for _ in range(n_samples) ])
        spIDs   = val[:,0]
        weights = val[:,1]
        
        # Sequence
        if self.setting.boolean.temporalModel:
            # sample-ID to idx
            imIDs = self.trainDataset.sampleID2imageID(spIDs)
            
            # Weights
            sequence_len = self.setting.general.sequence_len
            weights = [ w*np.ones(sequence_len) for w in weights ]
            weights = np.concatenate(weights)

        return spIDs.astype(int),imIDs.astype(int),weights 
        

    """ Validation metrics """
    def _metrics(self,measure,prediction):
        # Parameters
        max_steering = self.setting.preprocessing.maxSteering
        
        # Velocity regularization
        if self.speed_regularization:
            dev_speed_err = torch.abs(measure[ 'speed' ] - prediction[ 'speed' ])

        # if branch then reshape
        command    = measure   ['command']
        measure    = measure   ['actions']
        prediction = prediction['actions']
        if measure.size(-1) == 12:
            measure    = measure   .view(-1,4,3).sum(-2)
            prediction = prediction.view(-1,4,3).sum(-2)
        
        # Error
        dev_err  = torch.abs(measure - prediction)
        host_err = dev_err.data.cpu().numpy()

        # Error actions
        metrics = dict()
        metrics['SteerError'] = host_err[:,0] * max_steering
        metrics[  'GasError'] = host_err[:,1]
        metrics['BrakeError'] = host_err[:,2]
        if self.speed_regularization:
            metrics['SpeedError'] = dev_speed_err.data.cpu().numpy()
        
        # Measurements/Prediction
        metrics['Steer'    ] = measure   [:,0].data.cpu().numpy() * max_steering
        metrics['SteerPred'] = prediction[:,0].data.cpu().numpy() * max_steering

        if self.save_speed_action_plot:
            # Measurements
            metrics[  'Gas'] = measure[:,1].data.cpu().numpy()
            metrics['Brake'] = measure[:,2].data.cpu().numpy()

            # Prediction
            metrics[  'GasPred'] = prediction[:,1].data.cpu().numpy()
            metrics['BrakePred'] = prediction[:,2].data.cpu().numpy()

        # Mean
        steerMean = np.mean(metrics['SteerError'])
        gasMean   = np.mean(metrics[  'GasError'])
        brakeMean = np.mean(metrics['BrakeError'])
        metricsMean = [steerMean,gasMean,brakeMean]
        if self.speed_regularization:
            speedMean = np.mean(metrics['SpeedError'])
            metricsMean.append(speedMean)
        metricsMean = np.array(metricsMean)

        # Command control
        metrics['Command'] = command.data.cpu().numpy()

        return metrics,metricsMean


    def _stack(self,dataset,IDs):
        group = [ dataset[i] for i in IDs ]
        batch = {}
        for key in group[0]:
            batch[key] = torch.from_numpy( np.stack([data[key] for data in group]) )
        return batch
    
    def _transfer2device(self,batch):
        inputs    = ['frame','actions','speed','command','mask']
        dev_batch = {}

        for ko in inputs:
            if ko in batch:
                dev_batch[ko] = batch[ko].to(self.device)
        return dev_batch

    def _updatePriority(self,prediction,sampleID):
        # Losses to update
        losses = prediction['loss']
        losses = losses.view(-1,self.sequence_len).mean(1)
        losses = losses.data.cpu().numpy()

        # Update priority
        for idx,p in zip(sampleID,losses):
            self.samplePriority.update(idx,p)


    """ Exploration function
        --------------------
        Global train function
            *  Input: None
            * Output: total_loss (float) 
    """
    def _Exploration(self):
        # Parameters
        batch_size   = self.setting.general.batch_size
        sequence_len = self.setting.general.sequence_len
        stepView     = self.setting.general.stepView
        batch_size   = int(batch_size/sequence_len)

        # Loss
        running_loss = 0
        lossExp      = U.averager()

        # ID list
        spID,imID = self.trainDataset.generateIDs(False)
        loader = DataLoader(Dataset(self.trainDataset,imID),
                                    batch_size  = self.setting.general.batch_size,
                                    num_workers = self.init.num_workers,
                                    pin_memory  = True)

        # Exploration loop
        self.model.eval()
        with torch.no_grad(), tqdm(total=len(loader),leave=False) as pbar:
            for i, sample in enumerate(loader):
                # Batch
                batch,_ = sample
                dev_batch = self._transfer2device(batch)

                # Model
                dev_pred = self.model(dev_batch)
                dev_loss = self.lossEval(dev_batch,dev_pred)
                
                # Update priority
                self._updatePriority(dev_pred,spID[batch_size*i:batch_size*(i+1)])
                
                # Print statistics
                runtime_loss = dev_loss.item()
                running_loss += runtime_loss
                if i % stepView == (stepView-1):   # print every stepView mini-batches
                    message = 'BatchExplo loss=%.7f'
                    pbar.set_description( message % ( running_loss/stepView ))
                    pbar.refresh()
                    running_loss = 0.0
                lossExp.update(runtime_loss)
                pbar. update()
                pbar.refresh()
            pbar.close()

        if self.save_priority_history:
            pathPri = os.path.join( self._modelPath, "priority_init.pck" )
            self.samplePriority.save(pathPri)
            

    """ Train function
        --------------
        Global train function
            * Input: model     (nn.Module)
                     optimizer (torch.optim)
                     lossFunc  (function)
                     path      (path)
            * Output: total_loss (float) 
    """
    def _Train(self,epoch):
        # Parameters
        n_samples    = int(self.n_training*0.3)
        batch_size   = self.setting.general.batch_size
        sequence_len = self.setting.general.sequence_len
        stepView     = self.setting.general.stepView
        batch_size   = int(batch_size/sequence_len)

        # Loss
        running_loss = 0
        lossTrain    = U.averager()

        spIDs,imIDs,weights = self._samplingPrioritizedSamples(n_samples)
        loader = DataLoader(Dataset(self.trainDataset,imIDs,weights),
                                    batch_size  = self.setting.general.batch_size,
                                    num_workers = self.init.num_workers,
                                    pin_memory  = True)
        
        # Train loop
        self.model.train()
        with tqdm(total=len(loader),leave=False) as pbar:
            for i, sample in enumerate(loader):
                # Batch
                batch,_,weight = sample
                dev_batch = self._transfer2device(batch)

                # Model
                dev_pred = self.model(dev_batch)
                dev_loss = self.lossEval(dev_batch,dev_pred,weight)
                
                # Update priority
                self._updatePriority(dev_pred,spIDs[batch_size*i:batch_size*(i+1)])

                # zero the parameter gradients
                self.optimizer.zero_grad()
                self.model    .zero_grad()

                dev_loss.backward()#retain_graph=True)
                self.optimizer.step()
                
                # Print statistics
                runtime_loss = dev_loss.item()
                running_loss += runtime_loss
                if i % stepView == (stepView-1):   # print every stepView mini-batches
                    message = 'BatchTrain loss=%.7f'
                    pbar.set_description( message % ( running_loss/stepView ))
                    pbar.refresh()
                    running_loss = 0.0
                lossTrain.update(runtime_loss)
                pbar. update()
                pbar.refresh()
            pbar.close()

        lossTrain = lossTrain.val()
        print("Epoch training loss:",lossTrain)

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
        # Parameters
        stepView = self.setting.general.stepView

        # Loss
        running_loss = 0
        lossValid    = U.averager()
        
        # Metrics [Steer,Gas,Brake]
        avgMetrics = U.averager(3)
        metrics    = U.BigDict ( )

        # ID list
        imID = self.validDataset.generateIDs(True)
        loader = DataLoader(Dataset(self.validDataset,imID),
                                    batch_size  = self.setting.general.batch_size,
                                    num_workers = self.init.num_workers,
                                    pin_memory  = True)
        
        # Model to evaluation
        self.model.eval()
        with torch.no_grad(), tqdm(total=len(loader),leave=False) as pbar:
            for i, sample in enumerate(loader):
                # Batch
                batch,_ = sample
                dev_batch = self._transfer2device(batch)

                # Model
                dev_pred = self.model(dev_batch)
                dev_loss = self.lossEval(dev_batch,dev_pred)

                # Metrics
                metr, mean = self._metrics(dev_batch,dev_pred)
                metrics.update(metr)
                avgMetrics.update(mean)    

                # Calculate the loss
                runtime_loss  = dev_loss.item()
                running_loss += runtime_loss
                lossValid.update(runtime_loss)
                
                # Print statistics
                if i % stepView == (stepView-1):   # print every stepView mini-batches
                    message = 'BatchValid loss=%.7f'
                    pbar.set_description( message % ( running_loss/stepView ))
                    pbar.refresh()
                    running_loss = 0.0
                pbar. update()
                pbar.refresh()
            pbar.close()

        # Loss/metrics
        metrics      = metrics.resume()
        avgMetrics   = avgMetrics.mean
        running_loss = lossValid.val()
        
        # Print results
        print("Validation loss:",running_loss)
        info = "Steer: %0.5f\tGas: %0.5f\tBrake: %0.5f"%(avgMetrics[0],avgMetrics[1],avgMetrics[2])
        if self.speed_regularization: info += "\tSpeed: %0.5f"%avgMetrics[3]
        print(info)

        # Save figures
        SteerErrorPath = os.path.join(self._figureSteerErrorPath,"SteerErrorPath"+str(epoch)+".png")
        if self.save_speed_action_plot:
            GasErrorPath   = os.path.join(self._figureGasErrorPath  ,  "GasErrorPath"+str(epoch)+".png")
            BrakeErrorPath = os.path.join(self._figureBrakeErrorPath,"BrakeErrorPath"+str(epoch)+".png")
        
        F.saveColorMershError( metrics[ 'Steer' ], metrics['SteerError'],
                               metrics['Command'], SteerErrorPath,dom=(-1.20, 1.20))
        if self.save_speed_action_plot:
            F.saveHeatmap( metrics['Gas'  ], metrics[  'GasPred'], 'Gas',    GasErrorPath,range=[0,1] )
            F.saveHeatmap( metrics['Brake'], metrics['BrakePred'], 'Brake',BrakeErrorPath,range=[0,1] )

        return running_loss,avgMetrics
    

    """ Train/Evaluation """
    def execute(self,study=False):
        # Parameters
        n_epoch = self.setting.general.n_epoch
        
        # Study
        if study: return self.runStudy()

        # Initialize
        if self.init.is_loadedModel:
            valuesToSave = U.loadValuesToSave( os.path.join(self._modelPath,"model.csv") )
        else:
            valuesToSave = list()
        df = pd.DataFrame()
        tb = SummaryWriter('runs/'+self.setting.model+'/'+self.codename )

        # Exploration
        print("\nExploration")
        self._Exploration()

        # Loop over the dataset multiple times
        for epoch in range(self.epoch+1,n_epoch):
            print("\nEpoch",epoch,"-"*40)
            
            # Train
            lossTrain = self._Train(epoch)
            self.scheduler.step()
            self.samplePriority.step()
            
            # Validation
            lossValid,metr = self._Validation(epoch)
            
            # Save values metrics
            tb.add_scalar('Loss/Train', lossTrain, epoch)
            tb.add_scalar('Loss/Valid', lossValid, epoch)
            tb.add_scalar('MAE/Steer' , metr[0]  , epoch)
            tb.add_scalar('MAE/Gas'   , metr[1]  , epoch)
            tb.add_scalar('MAE/Brake' , metr[2]  , epoch)
            
            if self.speed_regularization: 
                tb.add_scalar('MAE/Speed',metr[3], epoch)
                valuesToSave.append( (lossTrain,lossValid,metr[0],metr[1],metr[2],metr[3]) )
                df = pd.DataFrame(valuesToSave, columns = ['LossTrain','LossValid','Steer','Gas','Brake','Speed'])
            else:
                valuesToSave.append( (lossTrain,lossValid,metr[0],metr[1],metr[2]) )
                df = pd.DataFrame(valuesToSave, columns = ['LossTrain','LossValid','Steer','Gas','Brake'])

            df.to_csv(self._modelPath + "/model.csv", index=False)

            # Save checkpoint
            if epoch%2 == 0:
                self._state_add (     'epoch',                    epoch  )
                self._state_add ('state_dict',self.    model.state_dict())
                self._state_add ( 'scheduler',self.scheduler.state_dict())
                self._state_add ( 'optimizer',self.optimizer.state_dict())
                self._state_save(epoch)


    """ Plot generate"""
    def plot(self,name):
        # Check paths
        self._checkFoldersToSave(name)
        paths = U.modelList(self._modelPath)
        
        # Initialize
        valuesToSave = list()
        df = pd.DataFrame()

        # Loop paths
        i = 0
        for epoch in range(len(paths)):
            epoch = epoch + 1
            print("\nEpoch",epoch,"-"*40)

            # Load
            checkpoint = torch.load(paths[i])
            self.model.load_state_dict(checkpoint['state_dict'])
            
            # Validation
            _,metr = self._Validation(epoch)
            
            # Save values metrics
            valuesToSave.append( (metr[0],metr[1],metr[2]) )
            df = pd.DataFrame(valuesToSave, columns = ['Steer','Gas','Brake'])
            i = i+1

            # Save metrics (csv)
            df.to_csv(self._modelPath + "/metrics.csv", index=False)  
    

    def _storeSignals(self,measure,prediction):
        # To CPU
        signal = {}
        signal['image'   ] =    measure[  'frame'  ]           .data.cpu().numpy()
        signal['command' ] =    measure[ 'command' ]           .data.cpu().numpy()
        signal['r_action'] =    measure[ 'actions' ]           .data.cpu().numpy()
        signal['alpha'   ] = prediction['attention'][ 'alpha' ].data.cpu().numpy()
        signal['beta'    ] = prediction['attention'][  'beta' ].data.cpu().numpy()
        signal['state'   ] = prediction[  'state'  ]           .data.cpu().numpy()
        signal['action'  ] = prediction[ 'actions' ]           .data.cpu().numpy()
        signal['hidden'  ] = prediction[  'signal' ][ 'hidden'].data.cpu().numpy()
        signal['feature' ] = prediction[  'signal' ]['feature'].data.cpu().numpy()
        signal['manager' ] = prediction[ 'manager' ]           .data.cpu().numpy()
        
        return signal


    """ Run study"""
    def runStudy(self):
        # Parameters
        umb = 20
        n   = 1

        # Loss
        signal = U.BigDict ( )

        # ID list
        imID = self.validDataset.generateIDs(True)
        loader = DataLoader(Dataset(self.validDataset,imID),
                                    batch_size  = self.setting.general.batch_size,
                                    num_workers = self.init.num_workers,
                                    pin_memory  = True)

        # Check paths
        studyPath = os.path.join(self._codePath ,'Study')
        U.checkdirectory(studyPath)

        # Load
        pathModel = os.path.join(self._modelPath,"model" + str(self.epoch) + ".pth" )
        checkpoint = torch.load(pathModel)
        self.model.load_state_dict(checkpoint['state_dict'])

        # Model to evaluation
        self.model.eval()
        print('Running study')
        with torch.no_grad(), tqdm(total=len(loader),leave=False) as pbar:
            for i, sample in enumerate(loader):
                # Batch
                batch,_ = sample
                dev_batch = self._transfer2device(batch)

                # Model
                dev_pred = self.model(dev_batch)
                host_s   = self._storeSignals(dev_batch,dev_pred)

                # Update
                host_s['id'] = np.array([i])
                signal.update(host_s)

                if i%umb == (umb-1):
                    # Resume
                    signal = signal.resume()

                    outfile = os.path.join(studyPath,'resume'+str(n)+'.sy')
                    with h5py.File(outfile,"w") as f:
                        dset = f.create_dataset('id'      , data=         i        )
                        dset = f.create_dataset('image'   , data=signal[   'image'])
                        dset = f.create_dataset('command' , data=signal[ 'command'])
                        dset = f.create_dataset('r_action', data=signal['r_action'])
                        dset = f.create_dataset('alpha'   , data=signal[   'alpha'])
                        dset = f.create_dataset('beta'    , data=signal[    'beta'])
                        dset = f.create_dataset('state'   , data=signal[   'state'])
                        dset = f.create_dataset('hidden'  , data=signal[  'hidden'])
                        dset = f.create_dataset('action'  , data=signal[  'action'])
                        dset = f.create_dataset('manager' , data=signal[ 'manager'])
                    signal = U.BigDict()
                    n += 1 

                pbar. update()
                pbar.refresh()
            pbar.close()
            
        """
        # t-SNE
        print("Execute t-SNE")
        for key in signals:
            print('Embedding '+ key)
            embedded = TSNE().fit_transform(signals[key]) # [n,2]
            path = os.path.join(self._modelPath,key+'.tsne')
            print('Save '+path+'\n')

            with open(path, 'wb') as handle:
                pickle.dump(embedded, handle, protocol=pickle.HIGHEST_PROTOCOL)
        """
                        
