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
    def __init__(self,):
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

    def cube(self,in_size=(92,196)):
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
        x = x.transpose(1, 2)                   # [batch,L,D]
        return x

""" Convolutional Neuronal Network - 5 layers
    -----------------------------------------
        * Input: img [batch,H,W]
        * Output: xt [batch,D,h,w]
"""
class CNN5Max(nn.Module):
    """ Constructor """
    def __init__(self,):
        super(CNN5Max, self).__init__()
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

    def cube(self,in_size=(92,196)):
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
        
        return x


""" ResNet 50
    ---------
    Ref: He, K., Zhang, X., Ren, S., & Sun, J. (2016). "Deep residual learning 
         for image recognition". In Proceedings of the IEEE conference on 
         computer vision and pattern recognition (pp. 770-778).
    
        * Input: img [batch,H,W]
        * Output: xt [batch,L,D]
"""
class ResNet50(nn.Module):
    """ Constructor """
    def __init__(self,):
        super(ResNet50, self).__init__()
        # Layers
        self.model = models.resnet50(pretrained=True)
        self.model = torch.nn.Sequential(*(list(self.model.children())[:-4]))
        self.linear1   = nn.Linear(512,256,bias=True)
        self.linear2   = nn.Linear(256,128,bias=True)
        self.linear3   = nn.Linear(128, 64,bias=True)
        self.LeakyReLu = nn.LeakyReLU()

        # Initialization
        torch.nn.init.xavier_uniform_(self.linear1.weight)
        torch.nn.init.xavier_uniform_(self.linear2.weight)
        torch.nn.init.xavier_uniform_(self.linear3.weight)
        
    def cube(self,in_size=(96,192)):
        x = int( in_size[0]/8 )
        y = int( in_size[1]/8 )
        return( x,y,64 )

    """ Forward """
    def forward(self,x):
        with torch.no_grad():
            x = self.model(x)                       # [batch,D,h,w]
            x = x.flatten(start_dim=2, end_dim=3)   # [batch,D,L]
            x = x.transpose(1, 2)                   # [batch,L,D]
        x = self.linear1(x.detach())
        x = self.LeakyReLu(x)
        x = self.linear2(x)
        x = self.LeakyReLu(x)
        x = self.linear3(x)
        x = self.LeakyReLu(x)
        return x

""" ResNet 50 (max)
    ---------------
        * Input: img [batch,H,W]
        * Output: xt [batch,D,h,w]
"""
class ResNet50Max(nn.Module):
    """ Constructor """
    def __init__(self,):
        super(ResNet50Max, self).__init__()
        # Layers
        self.model = models.resnet50(pretrained=True)
        self.model = torch.nn.Sequential(*(list(self.model.children())[:-4]))
        
    def cube(self,in_size=(96,192)):
        x = int( in_size[0]/8 )
        y = int( in_size[1]/8 )
        return( x,y,512 )

    """ Forward """
    def forward(self,x):
        with torch.no_grad():
            x = self.model(x)                       # [batch,D,h,w]
        return x


""" Wide residual network
    ---------------------
    Ref: He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning 
         for image recognition. In Proceedings of the IEEE conference on 
         computer vision and pattern recognition (pp. 770-778).
    
        * Input: img [batch,H,W]
        * Output: xt [batch,D,h,w]
"""
class WideResNet50(nn.Module):
    """ Constructor """
    def __init__(self,):
        super(WideResNet50, self).__init__()
        # Layers
        self.model     = models.wide_resnet50_2(pretrained=True)
        self.model     = torch.nn.Sequential(*(list(self.model.children())[:-4]))
        self.linear1   = nn.Linear(512,256,bias=True)
        self.linear2   = nn.Linear(256,128,bias=True)
        self.linear3   = nn.Linear(128, 64,bias=True)
        self.LeakyReLu = nn.LeakyReLU()

        # Initialization
        torch.nn.init.xavier_uniform_(self.linear1.weight)
        torch.nn.init.xavier_uniform_(self.linear2.weight)
        torch.nn.init.xavier_uniform_(self.linear3.weight)

    def cube(self,in_size=(96,192)):
        x = int( in_size[0]/8 )
        y = int( in_size[1]/8 )
        return( x,y,64 )

    """ Forward """
    def forward(self,x):
        with torch.no_grad():
            x = self.model(x)                       # [batch,D,h,w]
            x = x.flatten(start_dim=2, end_dim=3)   # [batch,D,L]
            x = x.transpose(1, 2)                   # [batch,L,D]
        x = self.linear1(x.detach())
        x = self.LeakyReLu(x)
        x = self.linear2(x)
        x = self.LeakyReLu(x)
        x = self.linear3(x)
        x = self.LeakyReLu(x)
        return x

""" Wide residual network (max)
    ---------------------------
    Ref: He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning 
         for image recognition. In Proceedings of the IEEE conference on 
         computer vision and pattern recognition (pp. 770-778).
    
        * Input: img [batch,H,W]
        * Output: xt [batch,D,h,w]
"""
class WideResNet50Max(nn.Module):
    """ Constructor """
    def __init__(self,):
        super(WideResNet50Max, self).__init__()
        # Layers
        self.model     = models.wide_resnet50_2(pretrained=True)
        self.model     = torch.nn.Sequential(*(list(self.model.children())[:-4]))

    def cube(self,in_size=(96,192)):
        x = int( in_size[0]/8 )
        y = int( in_size[1]/8 )
        return( x,y,512 )

    """ Forward """
    def forward(self,x):
        with torch.no_grad():
            x = self.model(x)                       # [batch,D,h,w]
        return x


class VGG19(nn.Module):
    """ Constructor """
    def __init__(self):
        super(VGG19, self).__init__()

        self.model     = models.vgg19_bn(pretrained=True)
        self.model     = torch.nn.Sequential(*(list(self.model.children())[:-4]))
        self.linear1   = nn.Linear(512,256,bias=True)
        self.linear2   = nn.Linear(256,128,bias=True)
        self.linear3   = nn.Linear(128, 64,bias=True)
        self.LeakyReLu = nn.LeakyReLU()

        # Initialization
        torch.nn.init.xavier_uniform_(self.linear1.weight)
        torch.nn.init.xavier_uniform_(self.linear2.weight)
        torch.nn.init.xavier_uniform_(self.linear3.weight)

    def cube(self,in_size=(96,192)):
        x = int( in_size[0]/8 )
        y = int( in_size[1]/8 )
        return( x,y,64 )

    """ Forward """
    def forward(self,x):
        with torch.no_grad():
            x = self.model(x)                       # [batch,D,h,w]
            x = x.flatten(start_dim=2, end_dim=3)   # [batch,D,L]
            x = x.transpose(1, 2)                   # [batch,L,D]
        if self.compression>0:
            x = self.linear1(x.detach())
            x = self.LeakyReLu(x)
        if self.compression>1:
            x = self.linear2(x)
            x = self.LeakyReLu(x)
        if self.compression>2:
            x = self.linear3(x)
            x = self.LeakyReLu(x)
        return x
        
