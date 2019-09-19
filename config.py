import random
import numpy as np
import torch
import json

_branchesList        = ['Codevilla18','Codevilla19']
_multimodalList      = ['Multimodal','Codevilla18','Codevilla19']
_speedRegressionList = ['Codevilla19']
_inputSpeedList      = ['Multimodal','Codevilla18','Codevilla19']
_outputSpeedList     = ['Codevilla19']
_temporalModelList   = ['Kim2017']

class BooleanConditions(object):
    def __init__(self,model):
        self.branches        = False    # Conditional (branches)
        self.multimodal      = False    # Multimodal (image + speed)
        self.inputSpeed      = False    # Input speed
        self.outputSpeed     = False    # Output speed
        self.speedRegression = False    # Speed regression
        
        self.temporalModel   = False

        self.branches        = model in _branchesList
        self.multimodal      = model in _multimodalList
        self.inputSpeed      = model in _inputSpeedList
        self.outputSpeed     = model in _outputSpeedList
        self.speedRegression = model in _speedRegressionList


class Init(object):
    def __init__(self):
        self.manual_seed =   False
        self.seed        =      -1
        self.device      =    None
        self.device_name = 'cuda:0'
        self.num_workers =       8
        
        self.is_loadedModel = False

        # Device
        self.device = torch.device(self.device_name)

    def set_seed(self,seed = -1):
        if seed < 0:
            self.seed = int(random.random()*1000000)
        else:
            self.seed = seed

        torch.cuda.manual_seed_all(self.seed)
        torch.     manual_seed    (self.seed)
        np.random        .seed    (self.seed)

    def device_(self,type_):
        self.device_name = type_
        self.device      = torch.device(type_)

    def load(self,path):
        self.manual_seed = True
        with open(path) as json_file:
            data = json.load(json_file)
            # Multitask
            self.num_workers = data["num_workers"]
            # Device
            self.device_(data["device_name"])
            # Seed
            self.set_seed(data["seed"])

    def print(self):
        pass

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

    def load(self,data):
        self.stepView       = data[      "stepView"]
        self.framePerFile   = data[  "framePerFile"]
        self.framePerSecond = data["framePerSecond"]
        
        self.validPath = data["validPath"]
        self.trainPath = data["trainPath"]
        self.savedPath = data["savedPath"]

    def print(self):
        print("\t"*2,"Train Path:\t\t"   ,self.trainPath)
        print("\t"*2,"Validation Path:\t",self.validPath)
        print("\t"*2,"Saved Path:\t\t"   ,self.savedPath)
        print("")

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

    def load(self,data):
        self.reshape       = data[      "reshape"]
        self.data_aug      = data[     "data_aug"]
        self.input_size    = data[   "input_size"]
        self.Laskey_noise  = data[ "Laskey_noise"]
        self.input_reshape = data["input_reshape"]

        self.max_steering = data["max_steering"]
        self.max_speed    = data[   "max_speed"]

    def print(self):
        print("\t"*2,"Data augmentation:\t",self.data_aug)
        print("\t"*2,"Laskey noise:\t\t"   ,self.Laskey_noise)
        print("\t"*2,"Input size:\t\t"     ,self.input_size)
        if self.reshape:
            print("\t"*2,"Reshape:\t"      ,self.input_reshape)
        print("\t"*2,"Maximum steering:\t",self.max_steering)
        print("\t"*2,"Maximum speed:\t\t" ,self.max_speed)
        print("")

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
        self.learning_rate_decay_steps  = 10
        self.learning_rate_decay_factor = 0.5

    def load(self,data):
        self.available = data["available"]

        self.learning_rate_initial      = data["learning_rate_initial"     ]
        self.learning_rate_decay_steps  = data["learning_rate_decay_steps" ]
        self.learning_rate_decay_factor = data["learning_rate_decay_factor"]

    def print(self):
        print("\t"*3,"Learning rate initial:\t\t"   , self.learning_rate_initial)
        print("\t"*3,"Learning rate decay factor:\t", self.learning_rate_decay_factor)
        print("\t"*3,"Learning rate decay steps:\t" , self.learning_rate_decay_steps)
        print("")

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
        self.beta_1        = 0.95   #0.9   #0.7 
        self.beta_2        = 0.999  #0.999 #0.85

    def load(self,data):
        self.type          = data[         "type"]
        self.beta_1        = data[       "beta_1"]
        self.beta_2        = data[       "beta_2"]
        self.learning_rate = data["learning_rate"]

    def print(self):
        print("\t"*3,"Learning rate:\t", self.learning_rate)
        print("\t"*3,"Beta 1:\t", self.beta_1)
        print("\t"*3,"Beta 2:\t", self.beta_2)
        print("")

    def save(self):
        return {
            "type"          : self.         type,
            "beta_1"        : self.       beta_1,
            "beta_2"        : self.       beta_2,
            "learning_rate" : self.learning_rate
        }


class _Loss_settings(object):
    def __init__(self):
        self.type          = "Weight"
        self.lambda_steer  = 0.45
        self.lambda_gas    = 0.45
        self.lambda_brake  = 0.10
        self.lambda_action = 0.95
        self.lambda_speed  = 0.05

    def load(self,data):
        self.type          = data[         "type"]
        self.lambda_gas    = data[   "lambda_gas"]
        self.lambda_steer  = data[ "lambda_steer"]
        self.lambda_brake  = data[ "lambda_brake"]
        self.lambda_speed  = data[ "lambda_speed"]
        self.lambda_action = data["lambda_action"]

    def print(self):
        if self.type is "Weight":
            print("\t"*3,"Lambda steer:\t" , self.lambda_steer)
            print("\t"*3,"Lambda gas:\t"   , self.lambda_gas)
            print("\t"*3,"Lambda brake:\t" , self.lambda_brake)
            print("\t"*3,"Lambda action:\t", self.lambda_action)
            print("\t"*3,"Lambda speed:\t" , self.lambda_speed)
            print("")

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

    def load(self,data):
        self.n_epoch      = data[     "n_epoch"]
        self.dropout      = data[     "dropout"]
        self.batch_size   = data[  "batch_size"]
        self.sequence_len = data["sequence_len"]

        self.scheduler.load(data["scheduler"])
        self.optimizer.load(data["optimizer"])
        self.loss     .load(data[     "loss"])

    def print(self):
        print("\t"*2,"Batch size:\t",  self.  batch_size)
        print("\t"*2,"Sequence len:\t",self.sequence_len)
        print("\t"*2,"Num. epochs:\t", self.     n_epoch)
        print("\t"*2,"Dropout:\t", self.dropout)
        print("")
        print("\t"*2,self.loss.type+" loss")
        self.loss.print()
        print("\t"*2,self.optimizer.type+" optimizer")
        self.optimizer.print()
        print("\t"*2,"Scheduler")
        self.scheduler.print()

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

    def load(self,data):
        self.metric = data["metric"]
    
    def print(self):
        print("\t"*2,"Metric:\t",self.metric)
        print("")

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

    def model_(self,model):
        self.model = model
        self.boolean = BooleanConditions(self.model)

    def load(self,path):
        with open(path) as json_file:
            data = json.load(json_file)

            self.model = data["model"]
            self.preprocessing.load(data["preprocessing"])
            self.evaluation   .load(data[   "evaluation"])
            self.general      .load(data[      "general"])
            self.train        .load(data[        "train"])

    def print(self):
        print("="*80)
        print("\n","\t"+self.model+" Model")
        print("\t"+"-"*64,"\n")

        print("\tGeneral")
        self.general.print()
        print("\tPreprocessing")
        self.preprocessing.print()
        print("\tTrain")
        self.train.print()
        print("\tEvaluation")
        self.evaluation.print()

        print("="*80)

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
            
