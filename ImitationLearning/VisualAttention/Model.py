import torch.nn as nn
from   torchvision import models
import common.directory as V
import ImitationLearning.VisualAttention.Decoder as D
import ImitationLearning.VisualAttention.Encoder as E

import ImitationLearning.VisualAttention.network.Attention      as A
import ImitationLearning.VisualAttention.network.Control        as C
import ImitationLearning.VisualAttention.network.Regularization as R

# Modules
def getModule(moduleList):
    module = {}
    _mod = moduleList
    for k in _mod:
        if   _mod[k] in V.Encoder  : module[  'Encoder'] = eval('E.'+_mod[  'Encoder'])
        elif _mod[k] in V.Decoder  : module[  'Decoder'] = eval('D.'+_mod[  'Decoder'])
        elif _mod[k] in V.Control  : module[  'Control'] = eval('C.'+_mod[  'Control'])
        elif _mod[k] in V.Attention: module['Attention'] = eval('A.'+_mod['Attention'])
        else : raise NameError('ERROR 404: module '+k+' no found')
    return module
    

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
    def __init__(self,setting):#(96,192)): 92,196
        super(Experimental, self).__init__()
        # Parameters
        in_dim   = setting.boolean.shape
        n_hidden = 1024
        module   = getModule(setting.modules)
        
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
    def __init__(self,setting):#(96,192)): 92,196
        super(ExpBranch, self).__init__()
        # Parameters
        in_dim   = setting.boolean.shape
        n_hidden = 1024
        module   = getModule(setting.modules)
        
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
                    

""" Approach
    --------
    ResNet    18   34   50
    depth1   128  128  512
    depth2   512  512 2048
"""
class Approach(nn.Module):
    """ Constructor """
    def __init__(self,setting, study = False):#(96,192)): 92,196
        super(Approach, self).__init__()
        # Setting
        encoder = setting.modules["Encoder"]
        if   encoder == 'ResNet34':
            # Encoder module
            self.lowEncoder = models.resnet34(pretrained=True)
            self.lowEncoder = nn.Sequential(*(list(self.lowEncoder.children())[:-4]))       
            HighEncoder     = E.HighResNet34() 

            # Parameters
            n_hidden  = 1024
            lowDepth  =  128
            highDepth =  512
            n_state   =   64

        elif encoder == 'ResNet50': 
            # Encoder module
            self.lowEncoder = models.resnet50(pretrained=True)
            self.lowEncoder = nn.Sequential(*(list(self.lowEncoder.children())[:-4]))       
            HighEncoder     = E.HighResNet50() 

            # Parameters
            n_hidden  = 2048
            lowDepth  =  512
            highDepth = 2048
            n_state   =  128

        else:
            print("ERROR: encoder no found (%s)"%encoder)
        
        if   setting.general.dataset == "CoRL2017": n_task = 2
        elif setting.general.dataset == "CARLA100": n_task = 3
        else:print("ERROR: dataset no found (%s)"%self.setting.general.dataset)
        vel_manager = True if n_task == 3 else False

        cube_dim    = (12,24,lowDepth) 
        n_encodeCmd =  16   # Length of code command control
        self.study  = study
        
        # Decoder
        cmdNet  = A.CommandNet(n_encodeCmd)                 # Command decoder
        spaAttn = A.SpatialAttnNet(cube_dim,n_state,study)  # Spatial attention
        ftrAttn = A.FeatureAttnNet(   highDepth,n_hidden,   # Feature attention
                                    n_encodeCmd,n_state,
                                                n_task,
                                                study)
        
        self.decoder = D.CatDecoder(HighEncoderNet = HighEncoder,
                                        SpatialNet = spaAttn,
                                        FeatureNet = ftrAttn,
                                        CommandNet = cmdNet,
                                    
                                       LowLevelDim =  lowDepth,
                                      HighLevelDim = highDepth,

                                          n_hidden = n_hidden,
                                          n_state  = n_state,
                                          n_task   = n_task,

                                             study = study)

        # Policy
        self.policy = C.MultiTaskPolicy(n_state,vel_manager)

        # Speed regularization
        self.SpeedReg = setting.train.loss.regularization
        if self.SpeedReg:
            self.regularization = R.SpeedRegModule(n_hidden)
        

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
        st,sig,attn = self.decoder(ηt,batch['mask'])
        at,ds = self.policy(st)
        
        # Regularization
        if self.SpeedReg: vt = self.regularization(sig['hidden'])
        else            : vt = None
        
        # State
        if not self.study:   st = None
        if not self.study:  sig = None
        if not self.study: attn = None

        return {  'actions' :   at,
                 'decision' :   ds,
                   'signal' :  sig,
                    'state' :   st,
                    'speed' :   vt,
                'attention' : attn}
                
