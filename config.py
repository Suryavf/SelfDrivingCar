import random
import numpy as np
import torch
import json

class Global(object):
    framePerSecond =  10
    num_workers    =   1
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
_inputSpeedList      = ['Multimodal','Codevilla18','Codevilla19']
_outputSpeedList     = ['Codevilla19']

class BooleanConditions(object):
    def __init__(self,model ):
        self.branches        = False
        self.multimodal      = False
        self.inputSpeed      = False
        self.outputSpeed     = False
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
        
        # Input speed
        if model in _inputSpeedList:
            self.inputSpeed = True
        else:
            self.inputSpeed = False

        # Output speed
        if model in _outputSpeedList:
            self.outputSpeed = True
        else:
            self.outputSpeed = False



class Init(object):
    def __init__(self):
        self.manual_seed =   False
        self.seed        =      -1
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

    def save(self,path):
        setting = {
            "manual_seed" : self.manual_seed,
            "device_name" : self.device_name,
            "num_workers" : self.num_workers,
            "seed"        : self.       seed,
        }

        with open(path, "w") as write_file:
            json.dump(setting, write_file, indent=4)


class General_settings(object):
    def __init__(self):
        self.framePerSecond =  10
        self.framePerFile   = 200
        self.stepView       =  10 # Print in Train

        # Path files
        self.validPath = "./data/h5file/SeqVal/"
        self.trainPath = "./data/h5file/SeqTrain/"
        self.savedPath = "/media/victor/Datos/Saved/"

    def save(self):
        return {
            "framePerSecond" : self.framePerSecond,
            "framePerFile"   : self.  framePerFile,
            "stepView"       : self.      stepView,

            "validPath" : self.validPath,
            "trainPath" : self.trainPath,
            "savedPath" : self.savedPath
        }


class Preprocessing_settings(object):
    def __init__(self):
        self.data_aug      = True
        self.Laskey_noise  = False
        self.input_size    = (88,200,3)
        self.input_reshape = (88,200)
        self.reshape       = False

        # Normalize
        self.max_speed    =  90
        self.max_steering = 1.2

    def save(self):
        return {
            "reshape"      : self.      reshape,
            "data_aug"     : self.     data_aug,
            "input_size"   : self.   input_size,
            "Laskey_noise" : self. Laskey_noise,

            "max_speed"    : self.   max_speed,
            "max_steering" : self.max_steering
        }


class _Scheduler_settings(object):
    def __init__(self):
        self.available = True

        self.learning_rate_initial      = 0.0001
        self.learning_rate_decay_steps  = 15
        self.learning_rate_decay_factor = 0.5

    def save(self):
        return {
            "learning_rate_decay_factor" : self.learning_rate_decay_factor,
            "learning_rate_decay_steps"  : self. learning_rate_decay_steps,
            "learning_rate_initial"      : self.     learning_rate_initial,
            "available"                  : self.                 available
        }


class _Optimizer_settings(object):
    def __init__(self):
        self.type          = "RAdam" # Adam, RAdam, Ranger
        self.learning_rate = 0.0001
        self.beta_1        = 0.70   #0.9  #0.7 
        self.beta_2        = 0.85   #0.999#0.85

    def save(self):
        return {
            "type"          : self.         type,
            "beta_1"        : self.       beta_1,
            "beta_2"        : self.       beta_2,
            "learning_rate" : self.learning_rate
        }


class _Loss_settings(object):
    def __init__(self):
        self.type          = "weight"
        self.lambda_steer  = 0.45
        self.lambda_gas    = 0.45
        self.lambda_brake  = 0.10
        self.lambda_action = 0.95
        self.lambda_speed  = 0.05

    def save(self):
        return {
            "type"          : self.         type,
            "lambda_gas"    : self.   lambda_gas,
            "lambda_steer"  : self. lambda_steer,
            "lambda_brake"  : self. lambda_brake,
            "lambda_speed"  : self. lambda_speed,
            "lambda_action" : self.lambda_action
        }


class Train_settings(object):
    def __init__(self):
        self.loss      =      _Loss_settings()
        self.optimizer = _Optimizer_settings()
        self.scheduler = _Scheduler_settings()

        self.n_epoch      = 150
        self.batch_size   = 120
        self.sequence_len =  20

        self.dropout = 0.5

    def save(self):
        return {
            "dropout"      : self.     dropout,
            "n_epoch"      : self.     n_epoch,
            "batch_size"   : self.  batch_size,
            "sequence_len" : self.sequence_len,
            
            "scheduler" : self.scheduler.save(),
            "optimizer" : self.optimizer.save(),
            "loss"      : self.     loss.save()
        }


class Evaluation_settings(object):
    def __init__(self):
        self.metric = "MAE"

    def save(self):
        return {
            "metric": self.metric
        }


class Setting(object):
    def __init__(self):
        self.preprocessing = Preprocessing_settings()
        self.evaluation    =    Evaluation_settings()
        self.general       =       General_settings()
        self.train         =         Train_settings()
        
        self.model   = "Kim2017" # Basic, Multimodal, Codevilla18, Codevilla19, Kim2017
        self.boolean = BooleanConditions(self.model)

    def save(self,path):
        setting = {
            "model": self.model,

            "preprocessing": self.preprocessing.save(),
            "evaluation"   : self.   evaluation.save(),
            "general"      : self.      general.save(),
            "train"        : self.        train.save()
        }

        with open(path, "w") as write_file:
            json.dump(setting, write_file, indent=4)
            
