import torch.nn as nn
import ImitationLearning.VisualAttention.Decoder as D
import ImitationLearning.VisualAttention.Encoder as E

import ImitationLearning.VisualAttention.network.Attention as attn

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
    def __init__(self):
        super(Experimental, self).__init__()
        # Modules
        self.attention = attn.Atten4()

        self.encoder = E.CNN5()
        self.decoder = D.BasicDecoder(self.attention)
    
    """ Forward """
    def forward(self,batch):
        x = self.encoder(batch['frame'])
        y = self.decoder(x)
        return {'actions': y}
        
