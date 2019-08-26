import random
import numpy as np
import torch

class Global(object):
    framePerSecond =  10
    num_workers    =   8
    stepView       =  10 # Print in Train
    max_steering   = 1.2
    max_speed      =  90

class Config(object):
    # Path files
    validPath = "./data/h5file/SeqVal/"
    trainPath = "./data/h5file/SeqTrain/"
    cookdPath = "/media/victor/Datos/Cooked"
    savedPath = "/media/victor/Datos/Saved"
    
    # Model
    model      = 'Codevilla19' # Basic, Multimodal, Codevilla18, Codevilla19, Kim2017
    n_epoch    =  80
    batch_size = 120
    
    # Learning rate
    learning_rate_initial      = 0.0001
    learning_rate_decay_steps  = 10
    learning_rate_decay_factor = 0.5

    # Adam Optimizer
    adam_lrate  = 0.0001
    adam_beta_1 = 0.7#0.9  #0.7 
    adam_beta_2 = 0.85#0.999#0.85

    # Loss
    lambda_steer = 0.45
    lambda_gas   = 0.45
    lambda_brake = 0.05
    lambda_action = 0.95
    lambda_speed  = 0.05
    

_branchesList        = ['Codevilla18','Codevilla19']
_multimodalList      = ['Multimodal','Codevilla18','Codevilla19']
_speedRegressionList = ['Codevilla19']

class BooleanConditions(object):
    def __init__(self,model ):
        self.branches        = False
        self.multimodal      = False
        self.speedRegression = False

        # Conditional (branches)
        if model in _branchesList:
            self.branches = True
        else:
            self.branches = False

        # Multimodal (image + speed)
        if model in _multimodalList:
            self.multimodal = True
        else:
            self.multimodal = False

        # Speed regression
        if model in _speedRegressionList:
            self.speedRegression = True
        else:
            self.speedRegression = False
        

class Init(object):
    def __init__(self):
        self.manual_seed =   False
        self.seed        =       0
        self.device      =    None
        self.device_name = 'cuda:0'
        self.num_workers =       8
        
        # Device
        self.device = torch.device(self.device_name)

    def set_seed(self,seed = -1):
        if seed < 0:
            self.seed = int(random.random()*1000000)
        else:
            self.seed = seed

        torch.manual_seed(self.seed)
        np.random   .seed(self.seed)

    def device_(self,type_):
        self.device_name = type_
        self.device = torch.device(type_)


class General_settings(object):
    def __init__(self):
        self.framePerSecond =  10
        self.framePerFile   = 200
        self.stepView       =  10 # Print in Train

        # Path files
        self.validPath = "./data/h5file/SeqVal/"
        self.trainPath = "./data/h5file/SeqTrain/"
        self.savedPath = "/media/victor/Datos/Saved"


class Preprocessing_settings(object):
    def __init__(self):
        self.data_aug     = True
        self.Laskey_noise = False
        self.input_size   = (88,200,3)
        self.reshape      = False

        # Normalize
        self.max_speed    =  90
        self.max_steering = 1.2


class _Scheduler_settings(object):
    def __init__(self):
        self.available = True

        self.learning_rate_initial      = 0.0001
        self.learning_rate_decay_steps  = 10
        self.learning_rate_decay_factor = 0.5


class _Optimizer_settings(object):
    def __init__(self):
        self.type          = "adam"
        self.learning_rate = 0.0001
        self.beta_1        = 0.70   #0.9  #0.7 
        self.beta_2        = 0.85   #0.999#0.85


class _Loss_settings(object):
    def __init__(self):
        self.type          = "weight"
        self.lambda_steer  = 0.45
        self.lambda_gas    = 0.45
        self.lambda_brake  = 0.10
        self.lambda_action = 0.95
        self.lambda_speed  = 0.05


class Train_settings(object):
    def __init__(self):
        self.loss      =      _Loss_settings()
        self.optimizer = _Optimizer_settings()
        self.scheduler = _Scheduler_settings()

        self.n_epoch      =  80
        self.batch_size   = 120
        self.sequence_len =  20

        self.dropout = 0.5


class Setting(object):
    def __init__(self):
        self.preprocessing = Preprocessing_settings()
        self.general       =       General_settings()
        self.train         =         Train_settings()
        
        self.model   = "Codevilla19"
        self.boolean = BooleanConditions(self.model)
        
