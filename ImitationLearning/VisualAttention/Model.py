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
    def __init__(self,shape=(92,196)):#(96,192)): 92,196
        super(Experimental, self).__init__()
        # Parameters
        in_dim   = shape
        n_hidden = 1024
        
        # Encoder
        self.encoder = E.CNN5()
        cube_dim = self.encoder.cube(in_dim)

        # Decoder
        self.attention = A.Atten9          (cube_dim,n_hidden)
        self.decoder   = D.TVADecoder2     (self.attention,cube_dim,n_hidden)
        self.control   = C.SumHiddenFeature(cube_dim,n_hidden)
    
    """ Forward """
    def forward(self,batch):
        x               = self.encoder(batch['frame'])
        x,hdn,attn,lstm = self.decoder(x)
        y               = self.control(x,hdn)
        return {  'actions' :     y,
                'attention' :  attn,
                   'hidden' : lstm}
                    
