import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from ImitationLearning.network.ResNet import resnet34 as ResNet


""" Xavier initialization
    ---------------------
    Args:
        model: torch model
"""
def xavierInit(model):
    if isinstance(model, nn.Linear):
        size = model.weight.size()
        fan_out = size[0] # number of rows
        fan_in  = size[1] # number of columns
        variance = np.sqrt(2.0/(fan_in + fan_out))
        model.weight.data.normal_(0.0, variance)


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

        return torch.squeeze(out)


""" Speed Regularization Module
    
    Fully connect network.
        * Input : torch (512)
        * Output: speed (scalar)

    Methods:
        @forward: Forward network
            - x: input
        @saveSettings: Save setting
            - path: directory to save

    Return: Name for directory model
"""
class SpeedRegModule(nn.Module):
    def __init__(self):
        super(SpeedRegModule, self).__init__()

        self._fully1 = nn.Linear(512,256)
        self._fully2 = nn.Linear(256,256)
        self._fully3 = nn.Linear(256, 1 )

    def forward(self,sig):
        h1 = F.relu(self._fully1(sig))
        h1 = F.dropout(h1, p=0.5, training=self.training)
        h2 = F.relu(self._fully2( h1))

        out = self._fully3(h2)

        return out
    

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

        self.fully1 = nn.Linear(512,256)
        self.fully2 = nn.Linear(256,256)
        self.fully3 = nn.Linear(256, 3 )

    def forward(self,sig):
        h1 = F.relu(self.fully1(sig))
        h1 = F.dropout(h1, p=0.5, training=self.training)

        h2 = F.relu(self.fully2( h1))

        out = self.fully3(h2)

        return out


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

        xavierInit(self._fully)
        xavierInit(self._out  )

    def forward(self,x):
        x = self._perception(x)
        x = F.dropout(x, p=0.5, training=self.training)

        x = self._fully(x)
        x = F.dropout(x, p=0.5, training=self.training)

        y_pred = self._out(x)

        return y_pred


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


""" Codevilla 2018 Network
    ----------------------
    Codevilla, F., Miiller, M., López, A., Koltun, V., & Dosovitskiy, A. (2018). 
    "End-to-end driving via conditional imitation learning". In 2018 IEEE International 
    Conference on Robotics and Automation (ICRA) (pp. 1-9).
    Ref: https://arxiv.org/pdf/1710.02410.pdf
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
class Codevilla18Net(nn.Module):
    def __init__(self):
        super(Codevilla18Net, self).__init__()

        self._perception    =        ResNet()
        self._measuredSpeed =   SpeedModule()
        self._jointLayer    = nn.Linear(640,512)

        self._branches = nn.ModuleList([ ControlModule() for i in range(4) ])

        # Inicialize
        self._perception   .apply(xavierInit)
        self._measuredSpeed.apply(xavierInit)
        xavierInit(self._jointLayer)
    
    def forward(self,img,vm,mask):
        percp = self._perception   (img)
        speed = self._measuredSpeed( vm)
        
        joint  = torch.cat( (percp,speed), dim=1  )
        signal = F.relu(self._jointLayer(joint))
        signal = F.dropout(signal, p=0.5, training=self.training)
        
        # Branches
        y_pred = torch.cat( [out(signal) for out in self._branches], dim=1)
        
        return y_pred*mask


""" Codevilla 2019 Network
    ----------------------
    Codevilla, F., Santana, E., López, A. M., & Gaidon, A. (2019). Exploring 
    the Limitations of Behavior Cloning for Autonomous Driving.
    Ref: https://arxiv.org/pdf/1904.08980.pdf
        * Input: image (matrix: 3,88,200)
                 speed (scalar)
        * Output: action (vector: 4) [Steer,Gas,Brake]
                  speed

    Methods:
        @forward: Forward network
            - img: image input
            - vm : speed input
        @saveSettings: Save setting
            - path: directory to save

    Return: Name for directory model
"""
class Codevilla19Net(nn.Module):
    def __init__(self):
        super(Codevilla19Net, self).__init__()

        self._perception    =           ResNet()
        self._measuredSpeed =      SpeedModule()
        self._regSpeed      =   SpeedRegModule()
        self._jointLayer    = nn.Linear(640,512)

        self._branches = nn.ModuleList([ ControlModule() for i in range(4) ])

        # Inicialize
        self._perception   .apply(xavierInit)
        self._measuredSpeed.apply(xavierInit)
        xavierInit(self._jointLayer)
    
    def forward(self,img,vm,mask):
        percp = self._perception   (img)
        speed = self._measuredSpeed( vm)
        
        # Actions prediction
        joint  = torch.cat( (percp,speed), dim=1  )
        signal = F.relu(self._jointLayer(joint))
        signal = F.dropout(signal, p=0.5, training=self.training)
        
        #Speed prediction
        v_pred = self._regSpeed(percp)
        
        # Branches
        y_pred = torch.cat( [out(signal) for out in self._branches], dim=1)
        y_pred = y_pred*mask
        
        
        return y_pred*mask,v_pred
            
