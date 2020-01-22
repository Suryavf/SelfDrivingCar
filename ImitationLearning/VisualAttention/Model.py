import torch.nn as nn
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
    def __init__(self,shape=(92,196)):#(96,192)):
        super(Experimental, self).__init__()
        # Parameters
        in_dim   = shape
        n_hidden = 1024
        
        # Encoder
        self.encoder = E.CNN5()
        cube_dim = self.encoder.cube(in_dim)

        # Decoder
        self.attention = A.Atten4          (cube_dim,n_hidden)
        self.control   = C.SumHiddenFeature(cube_dim,n_hidden)
        self.decoder   = D.DualDecoder(self.attention,self.control,cube_dim,n_hidden)
    
    """ Forward """
    def forward(self,batch):
        x              = self.encoder(batch['frame'])
        y,alpha,hidden = self.decoder(x)
        return {  'actions' :      y,
                'attention' :  alpha,
                   'hidden' : hidden}

