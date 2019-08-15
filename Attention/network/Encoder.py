import torch
import numpy as np
import torch.nn as nn

def xavierInit(model):
    if isinstance(model, nn.Conv2d):
        nn.init.xavier_uniform_(model.weight.data)
        model.bias.data.fill_(0)
    
    if isinstance(model, nn.Linear):
        size = model.weight.size()
        fan_out = size[0] # number of rows
        fan_in  = size[1] # number of columns
        variance = np.sqrt(2.0/(fan_in + fan_out))
        model.weight.data.normal_(0.0, variance)

""" Convolutional Neuronal Network - 5 layers
    -----------------------------------------
    Ref: Kim, J., & Canny, J. (2017). "Interpretable learning for self-driving 
         cars by visualizing causal attention". In Proceedings of the IEEE 
         international conference on computer vision (pp. 2942-2950).
    
        * Input: img [batch,h,w]
        * Output: xt [batch,L,D]
"""
class CNN5(nn.Module):
    """ Constructor """
    def __init__(self):
        super(CNN5, self).__init__()
        # Parameters
        self.D = 64 # cube_size[0]
        self.L = 90 # cube_size[1]*cube_size[2]
        self.R = 5760 # self.L*self.D
        self.H = 1024 # hidden_size
        self.M = 1024 # hidden_size
        self.sequence_len =  20
        self.batch_size   = 120
        
        # Layers
        self.net = nn.Sequential(
                        nn.Conv2d( 3, 24, kernel_size=5, stride=2),
                        nn.BatchNorm2d(24),
                        nn.ReLU(),
                        nn.Conv2d(24, 36, kernel_size=5, stride=2),
                        nn.BatchNorm2d(36),
                        nn.ReLU(),
                        nn.Conv2d(36, 48, kernel_size=3, stride=2),
                        nn.BatchNorm2d(48),
                        nn.ReLU(),
                        nn.Conv2d(48, 64, kernel_size=3, stride=1),
                        nn.BatchNorm2d(64),
                        nn.ReLU(),
                        nn.Conv2d(64, 64, kernel_size=3, stride=1),
                        nn.BatchNorm2d(64),
                        nn.ReLU()
                    )
        # Initialize
        self.net.apply(xavierInit)
        
    """ Forward """
    def forward(self,img):
        xt = self.net(img)                          # [batch,D,h,w]
        xt = xt.view(self.batch_size,self.D,self.L) # [batch,D,L]
        return xt.transpose(1, 2)                   # [batch,L,D]
    
