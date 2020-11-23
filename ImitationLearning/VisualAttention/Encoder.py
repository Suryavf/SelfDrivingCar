import torch
import torch.nn as nn
from   torchvision import models
from   ImitationLearning.backbone import EfficientNet
from   ImitationLearning.backbone import MemoryEfficientSwish
from   ImitationLearning.VisualAttention.network.Lambda import λLayer

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


""" ResNet 34
    ---------
    Ref: He, K., Zhang, X., Ren, S., & Sun, J. (2016). "Deep residual learning 
         for image recognition". In Proceedings of the IEEE conference on 
         computer vision and pattern recognition (pp. 770-778).
    
        * Input: img [batch,H,W]
        * Output: xt [batch,L,D]
"""
class ResNet34(nn.Module):
    """ Constructor """
    def __init__(self,):
        super(ResNet34, self).__init__()
        # Layers
        self.model = models.resnet34(pretrained=False)
        self.model = torch.nn.Sequential(*(list(self.model.children())[:-3]))
        
        self.convHead = nn.Conv2d(256,128, kernel_size=1, bias=False)
        self.bn       = nn.BatchNorm2d(num_features=128, momentum=0.99, eps=1e-3)
        self.swish    = MemoryEfficientSwish()

        # Initialization
        torch.nn.init.xavier_uniform_(self.convHead.weight)
        
    def cube(self,in_size=(96,192)):
        x = int( in_size[0]/16 )
        y = int( in_size[1]/16 )
        return( x,y,128 )

    """ Forward """
    def forward(self,x):
        x = self.model(x)                       # [batch,D,h,w]
        x = self.convHead(x)
        x = self.bn(x)
        x = self.swish(x)
        
        x = x.flatten(start_dim=2, end_dim=3)   # [batch,D,L]
        x = x.transpose(1, 2)                   # [batch,L,D]

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
        self.model = models.resnet50(pretrained=False)
        self.model = torch.nn.Sequential(*(list(self.model.children())[:-3]))
        
        self.convHead = nn.Conv2d(1024,128, kernel_size=1, bias=False)
        self.bn       = nn.BatchNorm2d(num_features=128, momentum=0.99, eps=1e-3)
        self.swish    = MemoryEfficientSwish()

        # Initialization
        torch.nn.init.xavier_uniform_(self.convHead.weight)
        
    def cube(self,in_size=(96,192)):
        x = int( in_size[0]/16 )
        y = int( in_size[1]/16 )
        return( x,y,128 )

    """ Forward """
    def forward(self,x):
        x = self.model(x)                       # [batch,D,h,w]
        x = self.convHead(x)
        x = self.bn(x)
        x = self.swish(x)
        
        x = x.flatten(start_dim=2, end_dim=3)   # [batch,D,L]
        x = x.transpose(1, 2)                   # [batch,L,D]

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
        

""" EfficientNet 
    ------------
    Ref: Mingxing Tan, Quoc V. Le (2019). EfficientNet: Rethinking Model 
         Scaling for Convolutional Neural Networks. In International 
         Conference on Machine Learning (pp. 6105-6114).

        * Input: img [batch,H,W]
        * Output: xt [batch,D,h,w]
"""
class EfficientNetB0(nn.Module):
    """ Constructor """
    def __init__(self,):
        super(EfficientNetB0, self).__init__()
        # Layers
        self.model = EfficientNet.from_name('efficientnet-b0')

    # (88,200) -> (96,192)
    def cube(self,in_size=(96,192)):
        x = int( in_size[0]/16 )
        y = int( in_size[1]/16 )
        return( x,y,128 )

    """ Forward """
    def forward(self,x):
        x = self.model(x)                       # [batch,D,h,w]
        x = x.flatten(start_dim=2, end_dim=3)   # [batch,D,L]
        x = x.transpose(1, 2)                   # [batch,L,D]
        return x
        

class EfficientNetB1(nn.Module):
    """ Constructor """
    def __init__(self,):
        super(EfficientNetB1, self).__init__()
        # Layers
        self.model = EfficientNet.from_name('efficientnet-b1')

    # (88,200) -> (96,192)
    def cube(self,in_size=(96,192)):
        x = int( in_size[0]/16 )
        y = int( in_size[1]/16 )
        return( x,y,128 )

    """ Forward """
    def forward(self,x):
        x = self.model(x)                       # [batch,D,h,w]
        x = x.flatten(start_dim=2, end_dim=3)   # [batch,D,L]
        x = x.transpose(1, 2)                   # [batch,L,D]
        return x
        

class EfficientNetB2(nn.Module):
    """ Constructor """
    def __init__(self,):
        super(EfficientNetB2, self).__init__()
        # Layers
        self.model = EfficientNet.from_name('efficientnet-b2')

    # (88,200) -> (96,192)
    def cube(self,in_size=(96,192)):
        x = int( in_size[0]/16 )
        y = int( in_size[1]/16 )
        return( x,y,128 )

    """ Forward """
    def forward(self,x):
        x = self.model(x)                       # [batch,D,h,w]
        x = x.flatten(start_dim=2, end_dim=3)   # [batch,D,L]
        x = x.transpose(1, 2)                   # [batch,L,D]
        return x
        

class EfficientNetB3(nn.Module):
    """ Constructor """
    def __init__(self,):
        super(EfficientNetB3, self).__init__()
        # Layers
        self.model = EfficientNet.from_name('efficientnet-b3')

    # (88,200) -> (96,192)
    def cube(self,in_size=(96,192)):
        x = int( in_size[0]/16 )
        y = int( in_size[1]/16 )
        return( x,y,128 )

    """ Forward """
    def forward(self,x):
        x = self.model(x)                       # [batch,D,h,w]
        x = x.flatten(start_dim=2, end_dim=3)   # [batch,D,L]
        x = x.transpose(1, 2)                   # [batch,L,D]
        return x


""" Lambda Networks 
    ---------------
    Ref: Anonymous (2021). LambdaNetworks: Modeling long-range 
         Interactions without Attention. ICLR 2021 Conference 
         Blind Submission.
         https://github.com/leaderj1001/LambdaNetworks/blob/main/model.py
"""
class λBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inDim, hdnDim, stride=1):
        super(λBottleneck, self).__init__()
        self.in1x1conv = nn.Conv2d(inDim, hdnDim, kernel_size=1, bias=False)
        self.BNorm1 = nn.BatchNorm2d(hdnDim)

        self.bottleneck = nn.ModuleList([λLayer(dim_in  = hdnDim, dim_out = hdnDim)])
        if stride != 1 or inDim != self.expansion * hdnDim:
            self.bottleneck.append(nn.AvgPool2d(kernel_size=(3, 3), stride=stride, padding=(1, 1)))
        self.bottleneck.append(nn.BatchNorm2d(hdnDim))
        self.bottleneck.append(nn.ReLU())
        self.bottleneck = nn.Sequential(*self.bottleneck)

        self.out_1x1conv = nn.Conv2d(hdnDim, self.expansion * hdnDim, kernel_size=1, bias=False)
        self.BNorm2 = nn.BatchNorm2d(self.expansion * hdnDim)
        
        self.ReLU = nn.ReLU()

        self.shortcut = nn.Sequential()
        if stride != 1 or inDim != self.expansion*hdnDim:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inDim, self.expansion*hdnDim, kernel_size=1, stride=stride),
                nn.BatchNorm2d(self.expansion*hdnDim)
            )

    def forward(self, fm):
        # Input
        x = self.in1x1conv(fm)
        x = self.   BNorm1(x)
        h = self.     ReLU(x)

        #Bottleneck
        h = self.bottleneck(h)

        # Output
        y = self.out_1x1conv(h)
        y = self.     BNorm2(y)

        # Skip connections
        y += self.shortcut(fm)
        return self.ReLU(y)


class λResNet(nn.Module):
    def __init__(self, block, n_block, mode='high'):
        super(λResNet, self).__init__()
        self.low  = (mode== 'low') | (mode=='total')
        self.high = (mode=='high') | (mode=='total')
        self.in_planes = 64
        
        if self.low:
            self.scell = nn.Sequential()
            self.scell.append(nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False))
            self.scell.append(nn.BatchNorm2d(64))
            self.scell.append(nn.ReLU(inplace=True))
            self.scell = nn.Sequential(*self.scell)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

            self.layer1 = self._make_layer(block,  64, n_block[0])
            self.layer2 = self._make_layer(block, 128, n_block[1], stride=2)
        
        if self.high:
            self.layer3 = self._make_layer(block, 256, n_block[2], stride=2)
            self.layer4 = self._make_layer(block, 512, n_block[3], stride=2)

            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Initialization
        for m in self.modules():    
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, n_hidden, num_blocks, stride=1):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, n_hidden, stride))
            self.in_planes = n_hidden * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.low:
            # Introduction
            x = self.  scell(x)
            x = self.maxpool(x)

            # Low level
            x = self.layer1(x)
            y = self.layer2(x)

        if self.high:
            y = self.layer3(y)
            y = self.layer4(y)
            y = self.avgpool(y)
        return y
        

def λResNet34(mode='total'):
    return λResNet(λBottleneck, [2, 2, 2, 2], mode)

def λResNet50(mode='total'):
    return λResNet(λBottleneck, [3, 4, 6, 3], mode)
    
