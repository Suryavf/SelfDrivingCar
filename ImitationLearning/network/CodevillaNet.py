import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from   torch.autograd import Variable as V
from   torch.utils.data import Dataset,DataLoader
from   torchvision import models, transforms, utils

from ImitationLearning.network.ResNet import resnet34        as ResNet
from common.data                      import CoRL2017Dataset as Dataset
from common.pytorch import xavierInit

from config import Config
from config import Global

# Settings
_global = Global()
_config = Config()

""" Basic Network for regression by ResNet34
    ----------------------------------------
    ResNet34 network.
        * Input: image (matrix: 3,88,200)
        * Output: action (vector: 1,3) [Steer,Gas,Brake]

    Methods:
        @forward: Forward network
            - x: image input
        @saveSettings: Save setting
            - path: directory to save

    Return: Name for directory model
"""
class BasicNet(nn.Module):
    def __init__(self):
        super(BasicNet, self).__init__()

        self._perception = ResNet()
        self._fully      = nn.Linear(512,256)
        self._out        = nn.Linear(256,  3)

    def forward(self,x):
        x = self._perception(x)
        x = F.dropout(x, p=0.5, training=self.training)

        x = self._fully(x)
        x = F.dropout(x, p=0.5, training=self.training)

        y_pred = self._out(x)

        return y_pred

    def saveSettings(self,path):
        setting = {
            "model"            : _config.            model,
            "n_epoch"          : _config.          n_epoch,
            "batch_size"       : _config.       batch_size,
            "time_demostration": _config.time_demostration,
            "Optimizer":{
                "type"         : "adam",
                "Learning_rate": {
                    "initial"     : _config.learning_rate_initial,
                    "decay_steps" : _config.learning_rate_decay_steps,
                    "decay_factor": _config.learning_rate_decay_factor
                },
                "beta_1": _config.adam_beta_1,
                "beta_2": _config.adam_beta_2
            },
            "Loss":{
                "type": "weighted",
                "Lambda":{
                    "steer": _config.lambda_steer,
                    "gas"  : _config.lambda_gas  ,
                    "brake": _config.lambda_brake
                }
            }
        }
        
        with open(path, "w") as write_file:
            json.dump(setting, write_file, indent=4)


""" Speed Module
    
    Fully connect network.
        * Input: speed measurement (scalar)
        * Output: torch (128)

    Methods:
        @forward: Forward network
            - x: input
        @saveSettings: Save setting
            - path: directory to save

    Return: Name for directory model
"""
class SpeedModule(nn.Module):
    def __init__(self):
        super(SpeedModule, self).__init__()

        self._fully1 = nn.Linear( 1 ,128)
        self._fully2 = nn.Linear(128,128)
        self._fully3 = nn.Linear(128,128)

    def forward(self,vm):
        h1 = F.relu(self._fully1(vm))
        h1 = F.dropout(h1, p=0.5, training=self.training)

        h2 = F.relu(self._fully2(h1))
        h2 = F.dropout(h2, p=0.5, training=self.training)

        out = F.relu(self._fully3(h2))
        out = F.dropout(out, p=0.5, training=self.training)

        return torch.squeeze(out) #out
  
    
""" Control Module
    
    Fully connect network.
        * Input : torch (512)
        * Output: action (vector: 1,3) [Steer,Gas,Brake]

    Methods:
        @forward: Forward network
            - x: input
        @saveSettings: Save setting
            - path: directory to save

    Return: Name for directory model
"""
class ControlModule(nn.Module):
    def __init__(self):
        super(ControlModule, self).__init__()

        self._fully1 = nn.Linear(512,256)
        self._fully2 = nn.Linear(256,256)
        self._fully3 = nn.Linear(256, 3 )

    def forward(self,sig):
        h1 = F.relu(self._fully1(sig))
        h1 = F.dropout(h1, p=0.5, training=self.training)

        h2 = F.relu(self._fully2( h1))
        #h2 = F.dropout(h2, p=0.5, training=self.training)

        out = self._fully3(h2)#F.tanh(self._fully3(h2))

        return out


""" Multimodal Network for regression
    ---------------------------------
    ResNet34 network.
        * Input: image (matrix: 3,88,200)
                 speed (scalar)
        * Output: action (vector: 3) [Steer,Gas,Brake]

    Methods:
        @forward: Forward network
            - img: image input
            - vm : speed input
        @saveSettings: Save setting
            - path: directory to save

    Return: Name for directory model
"""
class MultimodalNet(nn.Module):
    def __init__(self):
        super(MultimodalNet, self).__init__()

        self._perception    =        ResNet()
        self._measuredSpeed =   SpeedModule()
        self._control       = ControlModule()

        self._jointLayer    = nn.Linear(640,512)

        # Inicialize
        self._perception   .apply(xavierInit)
        self._measuredSpeed.apply(xavierInit)
        self._control      .apply(xavierInit)
        xavierInit(self._jointLayer)
    

    def forward(self,img,vm):
        percp = self._perception   (img)
        speed = self._measuredSpeed( vm)

        joint  = torch.cat( (percp,speed), dim=1  )
        signal = F.relu(self._jointLayer(joint))
        signal = F.dropout(signal, p=0.5, training=self.training)

        y_pred = self._control(signal)

        return y_pred

    def saveSettings(self,path):
        setting = {
            "model"            : _config.            model,
            "n_epoch"          : _config.          n_epoch,
            "batch_size"       : _config.       batch_size,
            "time_demostration": _config.time_demostration,
            "Optimizer":{
                "type"         : "adam",
                "Learning_rate": {
                    "initial"     : _config.learning_rate_initial,
                    "decay_steps" : _config.learning_rate_decay_steps,
                    "decay_factor": _config.learning_rate_decay_factor
                },
                "beta_1": _config.adam_beta_1,
                "beta_2": _config.adam_beta_2
            },
            "Loss":{
                "type": "weighted",
                "Lambda":{
                    "steer": _config.lambda_steer,
                    "gas"  : _config.lambda_gas  ,
                    "brake": _config.lambda_brake
                }
            }
        }
        
        with open(path, "w") as write_file:
            json.dump(setting, write_file, indent=4)

