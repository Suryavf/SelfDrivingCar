import torch
import torch.nn as nn
from   torchvision import models

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
        
        # Layers
        self.conv1 = nn.Conv2d( 3, 24, kernel_size=5, stride=2, bias=False)
        self.conv2 = nn.Conv2d(24, 36, kernel_size=5, stride=2, bias=False)
        self.conv3 = nn.Conv2d(36, 48, kernel_size=3, stride=2, bias=False)
        self.conv4 = nn.Conv2d(48, 64, kernel_size=3, stride=1, bias=False)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1, bias=False)

        self.batchN1 = nn.BatchNorm2d(24)
        self.batchN2 = nn.BatchNorm2d(36)
        self.batchN3 = nn.BatchNorm2d(48)
        self.batchN4 = nn.BatchNorm2d(64)
        self.batchN5 = nn.BatchNorm2d(64)
        
        self.ReLU = nn.ReLU()
        
        # Initialize
        torch.nn.init.xavier_uniform_(self.conv1.weight)
        torch.nn.init.xavier_uniform_(self.conv2.weight)
        torch.nn.init.xavier_uniform_(self.conv3.weight)
        torch.nn.init.xavier_uniform_(self.conv4.weight)
        torch.nn.init.xavier_uniform_(self.conv5.weight)
        self.batchN1.reset_parameters()
        self.batchN2.reset_parameters()
        self.batchN3.reset_parameters()
        self.batchN4.reset_parameters()
        self.batchN5.reset_parameters()

    def cube(self,in_size):
        x = int( (in_size[0]/4-13)/2 )
        y = int( (in_size[1]/4-13)/2 )
        return( x,y,64 )

    """ Forward """
    def forward(self,x):
        # Layer 1
        x = self.  conv1(x)
        x = self.batchN1(x)
        x = self.   ReLU(x)

        # Layer 2
        x = self.  conv2(x)
        x = self.batchN2(x)
        x = self.   ReLU(x)

        # Layer 3
        x = self.  conv3(x)
        x = self.batchN3(x)
        x = self.   ReLU(x)
        
        # Layer 4
        x = self.  conv4(x)
        x = self.batchN4(x)
        x = self.   ReLU(x)

        # Layer 5
        x = self.  conv5(x)
        x = self.batchN5(x)
        x = self.   ReLU(x)                     # [batch,D,h,w]
        
        x = x.flatten(start_dim=2, end_dim=3)   # [batch,D,L]
        return x.transpose(1, 2)                # [batch,L,D]
        

class _ResNet50():
    """ Constructor """
    def __init__(self):
        self.model     = models.resnet50(pretrained=True)
        self.model     = torch.nn.Sequential(*(list(self.model.children())[:-4]))
        self.model = self.model.to( torch.device('cuda:0') )

    def __call__(self, x): 
        with torch.no_grad():
            x = self.model(x)
            x = x.flatten(start_dim=2, end_dim=3)   # [batch,D,L]
            x = x.transpose(1, 2)                   # [batch,L,D]
            return x


class ResNet50(nn.Module):
    """ Constructor """
    def __init__(self):
        super(ResNet50, self).__init__()

        self.backbone = _ResNet50()
        #self.model     = models.resnet50(pretrained=True)
        #self.model     = torch.nn.Sequential(*(list(self.model.children())[:-4]))
        self.linear    = nn.Linear(512,64,bias= True)
        self.LeakyReLu = nn.LeakyReLU()

        # Initialization
        torch.nn.init.xavier_uniform_(self.linear.weight)

    def cube(self,in_size):
        x = int( in_size[0]/8 )
        y = int( in_size[1]/8 )
        return( x,y,64 )

    """ Forward """
    def forward(self,x):
        x = self.backbone(x)
        #with torch.no_grad():
        #    x = self.model(x)                       # [batch,D,h,w]
        #    x = x.flatten(start_dim=2, end_dim=3)   # [batch,D,L]
        #    x = x.transpose(1, 2)                   # [batch,L,D]
        x = self.linear(x.detach())
        return self.LeakyReLu(x)


class WideResNet50(nn.Module):
    """ Constructor """
    def __init__(self):
        super(WideResNet50, self).__init__()

        #self.model     = models.wide_resnet50_2(pretrained=True)
        #self.model     = torch.nn.Sequential(*(list(self.model.children())[:-4]))
        self.linear    = nn.Linear(512,64,bias= True)
        self.LeakyReLu = nn.LeakyReLU()

        # Initialization
        torch.nn.init.xavier_uniform_(self.linear.weight)

    def cube(self,in_size):
        x = int( in_size[0]/8 )
        y = int( in_size[1]/8 )
        return( x,y,64 )

    """ Forward """
    def forward(self,x):
        with torch.no_grad():
            x = self.model(x)                       # [batch,D,h,w]
            x = x.flatten(start_dim=2, end_dim=3)   # [batch,D,L]
            x = x.transpose(1, 2)                   # [batch,L,D]
        x = self.linear(x.detach())
        return self.LeakyReLu(x)


class VGG19(nn.Module):
    """ Constructor """
    def __init__(self):
        super(VGG19, self).__init__()

        self.model     = models.vgg19_bn(pretrained=True)
        self.model     = torch.nn.Sequential(*(list(self.model.children())[:-4]))
        self.linear    = nn.Linear(512,64,bias= True)
        self.LeakyReLu = nn.LeakyReLU()

        # Initialization
        torch.nn.init.xavier_uniform_(self.linear.weight)

    def cube(self,in_size):
        x = int( in_size[0]/8 )
        y = int( in_size[1]/8 )
        return( x,y,64 )

    """ Forward """
    def forward(self,x):
        with torch.no_grad():
            x = self.model(x)                       # [batch,D,h,w]
            x = x.flatten(start_dim=2, end_dim=3)   # [batch,D,L]
            x = x.transpose(1, 2)                   # [batch,L,D]
        x = self.linear(x)
        return self.LeakyReLu(x)
        
