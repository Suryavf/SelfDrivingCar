import json
import torch
import torch.nn as nn

import Attention.network.Decoder as deco
import Attention.network.Encoder as enco


""" Visual Attention Network
    ------------------------
    Ref: Kim, J., & Canny, J. (2017). "Interpretable learning for self-driving 
         cars by visualizing causal attention". In Proceedings of the IEEE 
         international conference on computer vision (pp. 2942-2950).
    
        * Input:     img [batch,h,w]
        * Output: action [batch,3]
"""
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

