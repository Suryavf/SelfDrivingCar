import torch.nn as nn
from   torchvision import models
import ImitationLearning.VisualAttention.Decoder as D
import ImitationLearning.VisualAttention.Encoder as E

import ImitationLearning.VisualAttention.network.Attention as A
import ImitationLearning.VisualAttention.network.Control   as C

""" Visual Attention Network
    ------------------------
    Ref: Kim, J., & Canny, J. (2017). "Interpretable learning for self-driving 
         cars by visualizing causal attention". In Proceedings of the IEEE 
         international conference on computer vision (pp. 2942-2950).
    
        * Input:     img [batch,h,w]
        * Output: action [batch,3]
"""
class Experimental(nn.Module):
    """ Constructor """
    def __init__(self,module,setting):#(96,192)): 92,196
        super(Experimental, self).__init__()
        # Parameters
        in_dim   = setting.boolean.shape
        n_hidden = 1024
        
        # Encoder
        self.encoder = module['Encoder']()
        cube_dim = self.encoder.cube(in_dim)

        # Decoder
        self.attention = module['Attention'](cube_dim,n_hidden)
        self.decoder   = module[  'Decoder'](self.attention,cube_dim,n_hidden)
        self.control   = module[  'Control'](cube_dim,n_hidden)
    
    """ Forward """
    def forward(self,batch):
        x               = self.encoder(batch['frame'])
        x,hdn,attn,lstm = self.decoder(x)
        y               = self.control(x,hdn)
        return {  'actions' :     y,
                'attention' :  attn,
                   'hidden' : lstm}
                    

class ExpBranch(nn.Module):
    """ Constructor """
    def __init__(self,module,setting):#(96,192)): 92,196
        super(ExpBranch, self).__init__()
        # Parameters
        in_dim   = setting.boolean.shape
        n_hidden = 1024
        
        # Encoder
        self.encoder = module['Encoder']()
        cube_dim = self.encoder.cube(in_dim)

        # Decoder
        self.attention = module['Attention'](cube_dim,n_hidden)
        self.decoder   = module[  'Decoder'](self.attention,cube_dim,n_hidden)
        self.control   = module[  'Control'](cube_dim,n_hidden)
    
    """ Forward """
    def forward(self,batch):
        x               = self.encoder(batch['frame'])
        x,hdn,attn,lstm = self.decoder(x)
        y               = self.control(x,hdn,batch['mask'])
        return {  'actions' :     y,
                'attention' :  attn,
                   'hidden' : lstm}
                    

class Approach(nn.Module):
    """ Constructor """
    def __init__(self,module,setting,n_hidden = 1024):#(96,192)): 92,196
        super(Approach, self).__init__()
        # Parameters
        self.H =     n_hidden       # 1024
        self.R = int(n_hidden/4)    #  256
        
        # ResNet            18   34   50
        depth1 = 128    #  128  128  512
        depth2 = 512    #  512  512 2048
        cube_dim = (12,24,depth1)
        n_command = 16

        # Encoder
        self.lowEncoder = models.resnet18(pretrained=True)
        self.lowEncoder = nn.Sequential(*(list(self.lowEncoder.children())[:-4]))       
        HighEncoder = E.λResNet34(cube_dim,'high')
        
        # Decoder
        cmdNet  = A.CommandNet(n_command)                   # Command decoder
        spaAttn = A.SpatialAttnNet(cube_dim)                # Spatial attention
        ftrAttn = A.FeatureAttnNet(depth2,self.H,n_command) # Feature attention
        self.decoder = D.CatDecoder(HighEncoder,spaAttn,ftrAttn,cmdNet)

        # Policy
        self.policy = C.Policy(64)
        

    """ Backbone 
    
        ResNet    18   34   50
        ----------------------
        depth1   128  128  512
        depth2   512  512 2048
    """
    def backbone(self,name):
        if   name == 'resnet18':
            return models.resnet18(pretrained=True), (128, 512)
        elif name == 'resnet34':
            return models.resnet34(pretrained=True), (128, 512)
        elif name == 'resnet50':
            return models.resnet50(pretrained=True), (512,2048)
        

    """ Forward """
    def forward(self,batch):
        # Visual encoder
        ηt = self.lowEncoder(batch['frame'])
        st = self.decoder(ηt,batch['mask'])
        at = self.policy(st)

        return {  'actions' :    at},
                #'attention' :  None,
                #   'hidden' :  None}

