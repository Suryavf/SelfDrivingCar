import random
import json
import numpy as np
import torch

class Setting(object):
    def __init__(self):
        self.preprocessing = Preprocessing_settings()
        self.evaluation    =    Evaluation_settings()
        self.sampling      =      Sampling_settings()
        self.general       =       General_settings()
        self.train         =         Train_settings()
        
        self.model   =  "ExpBranch"        # Basic, Multimodal, Codevilla18, Codevilla19, Kim2017, Experimental, ExpBranch
        self.modules = {
                        "Encoder"   : "CNN5"      ,      # CNN5, ResNet50, WideResNet50, VGG19, EfficientNetB0-3
                        "Decoder"   : "TVADecoder",      # BasicDecoder, DualDecoder, TVADecoder
                        "Attention" : "Atten11"   ,      # Atten1-12
                        "Control"   : "BranchesModule"   # SumHiddenFeature,BranchesModule
                        }
        
        self.boolean = BooleanConditions(self.model,self.modules,self.train)
        
    def model_(self,model):
        self.model = model
        self.boolean = BooleanConditions(self.model,self.modules,self.train)

    def load(self,path):
        with open(path) as json_file:
            data = json.load(json_file)
            if "model" in data:
                self.model   = data[  "model"]
            if "modules" in data:
                modul = data["modules"]
                if "Encoder" in modul:
                    self.modules["Encoder"  ] = modul[  "Encoder"]
                if "Decoder" in modul:
                    self.modules["Decoder"  ] = modul[  "Decoder"]
                if "Attention" in modul:
                    self.modules["Attention"] = modul["Attention"]
                if "Control" in modul:
                    self.modules["Control"  ] = modul[  "Control"]
            if "preprocessing" in data:
                self.preprocessing.load(data["preprocessing"])
            if "evaluation" in data:
                self.evaluation   .load(data[   "evaluation"])
            if "sampling" in data:
                self.sampling     .load(data[     "sampling"])
            if "general" in data:
                self.general      .load(data[      "general"])
            if "train" in data:
                self.train        .load(data[        "train"])

            self.boolean = BooleanConditions(self.model,self.modules,self.train)

    def print(self):
        print("="*80,"\n\n","\t"+self.model+" Model\n","\t"+"-"*64,"\n")
        print("\tGeneral")
        self.general.print()
        print("\tPriorized sampling")
        self.sampling.print()
        print("\tTrain")
        self.train.print()
        print("="*80) 
        
    def save(self,path):
        setting = {
            "model"  : self.model,
            "modules": self.modules,

            "preprocessing": self.preprocessing.save(),
            "evaluation"   : self.   evaluation.save(),
            "sampling"     : self.     sampling.save(),
            "general"      : self.      general.save(),
            "train"        : self.        train.save()
        }

        with open(path, "w") as write_file:
            json.dump(setting, write_file, indent=4)
            

class Init(object):
    def __init__(self):
        self.manual_seed =   False
        self.seed        =      -1
        self.device      =    None
        self.device_name = 'cuda:0'
        self.num_workers =       4
        
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
            if "num_workers" in data:
                self.num_workers = data["num_workers"]
            # Device
            if "device_name" in data:
                self.device_(data["device_name"])
            # Seed
            if "seed" in data:
                self.set_seed(data["seed"])
            else:
                self.set_seed()

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
        self.stepView       =   5 # Print in Train

        self.n_epoch      = 400
        self.batch_size   = 120
        self.sequence_len =  20

        self.slidingWindow  = 5

        # Path files
        self.dataset   = "CoRL2017" # CoRL2017 CARLA100
        self.validPath = "./data/h5file/SeqVal/"
        self.trainPath = "./data/h5file/SeqTrain/"
        self.savedPath = "/media/victor/Datos/Saved/"

    def load(self,data):
        if "stepView" in data:
            self.stepView       = data[      "stepView"]
        if "framePerFile" in data:
            self.framePerFile   = data[  "framePerFile"]
        if "framePerSecond" in data:
            self.framePerSecond = data["framePerSecond"]
        
        if "n_epoch" in data:
            self.n_epoch      = data[     "n_epoch"]
        if "batch_size" in data:
            self.batch_size   = data[  "batch_size"]
        if "sequence_len" in data:
            self.sequence_len = data["sequence_len"]

        if "slidingWindow" in data:
            self.slidingWindow = data["slidingWindow"]

        if "dataset" in data:
            self.dataset   = data[  "dataset"]
        if "validPath" in data:
            self.validPath = data["validPath"]
        if "trainPath" in data:
            self.trainPath = data["trainPath"]
        if "savedPath" in data:
            self.savedPath = data["savedPath"]

    def print(self):
        print("\t"*2,"Train Path:\t\t"   ,self.trainPath)
        print("\t"*2,"Validation Path:\t",self.validPath)
        print("\t"*2,"Saved Path:\t\t"   ,self.savedPath)
        print("")
        print("\t"*2,"Batch size:\t",  self.  batch_size)
        print("\t"*2,"Sequence len:\t",self.sequence_len)
        print("\t"*2,"Num. epochs:\t", self.     n_epoch)
        print("")

    def save(self):
        return {
            "framePerSecond" : self.framePerSecond,
            "framePerFile"   : self.  framePerFile,
            "stepView"       : self.      stepView,
            
            "n_epoch"      : self.     n_epoch,
            "batch_size"   : self.  batch_size,
            "sequence_len" : self.sequence_len,
            
            "dataset"   : self.  dataset,
            "validPath" : self.validPath,
            "trainPath" : self.trainPath,
            "savedPath" : self.savedPath
        }


class Sampling_settings(object):
    def __init__(self):
        self.alpha = 1.0
        self. beta = 0.0

        # Beta function
        self.betaLinear = True
        self.betaPhase  = 50 # epochs

        # Upper Confidence Bound 1 applied to trees (UCT)
        self.balance = True
        self.c       = 5.0

        
    def load(self,data):
        if "alpha" in data:
            self.alpha = data["alpha"]
        if "beta" in data:
            self.beta  = data[ "beta"]
        if "betaLinear" in data:
            self.betaLinear = data["betaLinear"]
        if "betaPhase" in data:
            self.betaPhase = data[ "betaPhase"]
        if "balance" in data:
            self.balance = data["balance"]
        if "c" in data:
            self.c   = data[ "c" ]

    def print(self):
        print("\t"*2,"Alpha:\t",self.alpha)
        if self.betaLinear: print("\t"*2,"Beta:\t",self.beta,' (Phase:',self.betaPhase,')')
        else              : print("\t"*2,"Beta:\t",self.beta)
        if self.balance:
            print("\t"*2,"Use balance (UCT, c=%1.1f)"%self.c)
        print("")

    def save(self):
        return {
            "alpha" : self.alpha,
            "beta"  : self. beta,
            "betaLinear": self.betaLinear,
            "betaPhase" : self.betaPhase,
            "balance": self.balance,
            "c": self.c
        }


class Preprocessing_settings(object):
    def __init__(self):
        self.dataAug      = True
        self.LaskeyNoise  = False
        
        # Normalize
        self.maxSpeed    =  90
        self.maxSteering = 1.2

    def load(self,data):
        if "dataAug" in data:
            self.dataAug     = data[    "dataAug"]
        if "LaskeyNoise" in data:
            self.LaskeyNoise = data["LaskeyNoise"]

        if "maxSteering" in data:
            self.maxSteering = data["maxSteering"]
        if "maxSpeed" in data:
            self.maxSpeed    = data[   "maxSpeed"]

    def print(self):
        print("\t"*2,"Data augmentation:\t",self.dataAug)
        print("\t"*2,"Laskey noise:\t\t"   ,self.LaskeyNoise)
        print("\t"*2,"Maximum steering:\t" ,self.maxSteering)
        print("\t"*2,"Maximum speed:\t\t"  ,self.maxSpeed)
        print("")

    def save(self):
        return {
            "data_aug"     : self.     dataAug,
            "Laskey_noise" : self. LaskeyNoise,

            "max_speed"    : self.   maxSpeed,
            "max_steering" : self.maxSteering
        }


class _Scheduler_settings(object):
    def __init__(self):
        self.available = True

        self.learning_rate_decay_steps  =  20
        self.learning_rate_decay_factor = 0.5

    def load(self,data):
        if "available" in data:
            self.available = data["available"]

        if "learning_rate_decay_steps" in data:
            self.learning_rate_decay_steps  = data["learning_rate_decay_steps" ]
        if "learning_rate_decay_factor" in data:
            self.learning_rate_decay_factor = data["learning_rate_decay_factor"]

    def print(self):
        print("\t"*3,"Learning rate decay factor:\t", self.learning_rate_decay_factor)
        print("\t"*3,"Learning rate decay steps:\t" , self.learning_rate_decay_steps)
        print("")

    def save(self):
        return {
            "learning_rate_decay_factor" : self.learning_rate_decay_factor,
            "learning_rate_decay_steps"  : self. learning_rate_decay_steps,
            "available"                  : self.                 available
        }


class _Optimizer_settings(object):
    def __init__(self):
        self.type         = "Adam" # Adam, RAdam, Ranger
        self.learningRate = 0.0001
        self.beta1        = 0.70  #0.9   #0.7 
        self.beta2        = 0.85  #0.999 #0.85

    def load(self,data):
        if "type" in data:
            self.type         = data[        "type"]
        if "beta1" in data:
            self.beta1        = data[       "beta1"]
        if "beta2" in data:
            self.beta2        = data[       "beta2"]
        if "learningRate" in data:
            self.learningRate = data["learningRate"]

    def print(self):
        print("\t"*3,"Learning rate:\t", self.learningRate)
        print("\t"*3,'Beta:\t(%1.2f,%1.2f)' % (self.beta1,self.beta2))
        print("")

    def save(self):
        return {
            "type"         : self.       type,
            "beta1"        : self.      beta1,
            "beta2"        : self.      beta2,
            "learningRate" : self.learningRate
        }


class _Loss_settings(object):
    def __init__(self):
        self.type           = "Weighted" # Weighted, WeightedReg, WeightedMultiTask
        self.regularization = True
        self.lambda_steer   = 0.45
        self.lambda_gas     = 0.45
        self.lambda_brake   = 0.10
        self.lambda_desc    = 0.00
        self.lambda_action  = 0.95
        self.lambda_speed   = 0.05
        """
        "lambda_gas"  : 0.33333,
            "lambda_steer": 0.26667,
            "lambda_brake": 0.06667,
            "lambda_desc" : 0.33333,
            "lambda_speed" : 0.05,
            "lambda_action": 0.95
        """

    def load(self,data):
        if "type" in data:
            self.type           = data[          "type"]
        if "regularization" in data:
            self.regularization = data["regularization"]
        if "lambda_gas" in data:
            self.lambda_gas     = data[    "lambda_gas"]
        if "lambda_steer" in data:
            self.lambda_steer   = data[  "lambda_steer"]
        if "lambda_brake" in data:
            self.lambda_brake   = data[  "lambda_brake"]
        if "lambda_desc" in data:
            self.lambda_desc    = data[   "lambda_desc"]
        if "lambda_speed" in data:
            self.lambda_speed   = data[  "lambda_speed"]
        if "lambda_action" in data:
            self.lambda_action  = data[ "lambda_action"]

    def print(self):
        if self.type == "Weighted":
            print("\t"*3,"Lambda steer:\t" , self.lambda_steer )
            print("\t"*3,"Lambda gas:\t"   , self.lambda_gas   )
            print("\t"*3,"Lambda brake:\t" , self.lambda_brake )
            print("")
        elif self.type == "WeightedReg":
            print("\t"*3,"Lambda steer:\t" , self.lambda_steer )
            print("\t"*3,"Lambda gas:\t"   , self.lambda_gas   )
            print("\t"*3,"Lambda brake:\t" , self.lambda_brake )
            print("")
            print("\t"*3,"Lambda action:\t", self.lambda_action)
            print("\t"*3,"Lambda speed:\t" , self.lambda_speed )
            print("")
        elif self.type == "WeightedMultiTask":
            print("\t"*3,"Lambda steer:\t"   , self.lambda_steer )
            print("\t"*3,"Lambda gas:\t"     , self.lambda_gas   )
            print("\t"*3,"Lambda brake:\t"   , self.lambda_brake )
            print("\t"*3,"Lambda decision:\t", self.lambda_desc  )
            print("")
            print("\t"*3,"Lambda action:\t", self.lambda_action)
            print("\t"*3,"Lambda speed:\t" , self.lambda_speed )
            print("")

    def save(self):
        return {
            "type"          : self.          type,
            "regularization": self.regularization,
            "lambda_gas"    : self.    lambda_gas,
            "lambda_steer"  : self.  lambda_steer,
            "lambda_brake"  : self.  lambda_brake,
            "lambda_desc"   : self.   lambda_desc,
            "lambda_speed"  : self.  lambda_speed,
            "lambda_action" : self. lambda_action
        }


class Train_settings(object):
    def __init__(self):
        self.loss      =      _Loss_settings()
        self.optimizer = _Optimizer_settings()
        self.scheduler = _Scheduler_settings()

        self.dropout = 0.5

    def load(self,data):
        if "scheduler" in data:
            self.scheduler.load(data["scheduler"])
        if "optimizer" in data:
            self.optimizer.load(data["optimizer"])
        if "loss" in data:
            self.loss     .load(data[     "loss"])

        if "dropout" in data:
            self.dropout   = data[     "dropout"]

    def print(self):
        # print("\t"*2,"Dropout:\t", self.dropout)
        # print("")
        print("\t"*2,self.loss.type+" loss")
        self.loss.print()
        print("\t"*2,self.optimizer.type+" optimizer")
        self.optimizer.print()
        print("\t"*2,"Scheduler")
        self.scheduler.print()

    def save(self):
        return {
            "dropout"      : self.     dropout,
            
            "scheduler" : self.scheduler.save(),
            "optimizer" : self.optimizer.save(),
            "loss"      : self.     loss.save()
        }


class Evaluation_settings(object):
    def __init__(self):
        self.metric = "MAE"

    def load(self,data):
        if "metric" in data:
            self.metric = data["metric"]
    
    def print(self):
        print("\t"*2,"Metric:\t",self.metric)
        print("")

    def save(self):
        return {
            "metric": self.metric
        }

"""
    Boolean conditions
    ------------------
"""
# Boolean conditions (model, no approach)
_branchesList        = ['Codevilla18','Codevilla19']
_speedRegressionList = ['Codevilla19']
_inputSpeedList      = ['Multimodal','Codevilla18','Codevilla19']
_outputSpeedList     = ['Codevilla19']
_temporalModelList   = ['Kim2017']

# Boolean conditions (approach modules)
_modularModel       = ['Experimental','ExpBranch','Approach']
_branchesModList    = ['BranchesModule']
_temporalModList    = ['BasicDecoder', 'DualDecoder', 'TVADecoder','CatDecoder']
_inputSpeedModList  = [] # Control
_outputSpeedModList = ['WeightedReg','WeightedMultiTask']

_shape = {  'CNN5'          : (92,196), 
            'ResNet34'      : (96,192),
            'ResNet50'      : (96,192), 
            'WideResNet50'  : (96,192), 
            'VGG19'         : (96,192), 
            'EfficientNetB0': (96,192), 
            'EfficientNetB1': (96,192), 
            'EfficientNetB2': (96,192), 
            'EfficientNetB3': (96,192)
          }
class BooleanConditions(object):
    def __init__(self,model,modules,train):
        self.branches        = False    # Conditional (branches)
        self.temporalModel   = False
        self.inputSpeed      = False    # Input speed
        self.outputSpeed     = False    # Output speed
        
        # Branches
        _model   =  model in _branchesList
        _modular = (model in _modularModel) and (modules['Control'] in _branchesModList)
        self.branches = _model or _modular

        # Temporal
        _model   =  model in _temporalModelList
        _modular = (model in _modularModel) and (modules['Decoder'] in _temporalModList)
        self.temporalModel = _model or _modular

        # Input speed
        _model   =  model in _inputSpeedList
        _modular = (model in _modularModel) and (modules['Control'] in _inputSpeedModList)
        self.inputSpeed = _model or _modular

        # Output speed
        _model   =  model in _outputSpeedList
        _modular = (model in _modularModel) and (train.loss.type in _outputSpeedModList)
        self.outputSpeed = _model or _modular
        
        self.backbone = modules['Encoder']
        self.shape    = _shape[self.backbone]
        
