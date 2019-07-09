import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from   torchvision import models, transforms, utils
from   torch.utils.data import Dataset,DataLoader
from   torch.autograd import Variable as V

import numpy as np

from ImitationLearning.network.ResNet import resnet34        as ResNet
from common.data                      import CoRL2017Dataset as Dataset
from config                           import Config

def weights_init(m):
    if isinstance(m, nn.Linear):
        size = m.weight.size()
        fan_out = size[0] # number of rows
        fan_in  = size[1] # number of columns
        variance = np.sqrt(2.0/(fan_in + fan_out))
        m.weight.data.normal_(0.0, variance)

class ResNetReg(nn.Module):
    def __init__(self):
        super(ResNetReg, self).__init__()

        self._perception = ResNet()
        self._fully      = nn.Linear(512,256)
        self._out        = nn.Linear(256,  3)

    def forward(self,x):
        percep =        self._perception(x)
        percep = F.dropout(percep, p=0.5, training=self.training)

        hidden = F.relu(self._fully(percep))
        hidden = F.dropout(hidden, p=0.5, training=self.training)

        y_pred = self._out  (hidden)

        return y_pred
