import json
import torch
import torch.nn as nn

import Attention.Network.Decoder as deco
import Attention.Network.Encoder as enco

from config import Config
from config import Global

# Settings
_global = Global()
_config = Config()

class Kim2017Net(nn.Module):
    """ Constructor """
    def __init__(self):
        super(Kim2017Net, self).__init__()

        # Modules
        self.encoder = enco.CNN5()
        self.decoder = deco.Kim2017(3)
    
    """ Forward """
    def forward(self,img):
        x = self.encoder(img)
        y = self.decoder(x)
        return y
    
    """ Save settings """
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
            
