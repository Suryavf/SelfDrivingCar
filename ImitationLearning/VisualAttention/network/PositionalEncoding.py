import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

""" Basic Sinusoids
    ---------------
    Ref: Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, 
         A. N., Lukasz Kaiser & Polosukhin, I. (2017). "Attention is all you need". 
         In Advances in neural information processing systems (pp. 5998-6008).
 """
class Vaswani2017(nn.Module):
    """ Constructor """
    def __init__(self, cube_size):
        self.D = cube_size[2]  
        self.H = cube_size[1]
        self.W = cube_size[0]
        self.L = cube_size[0]*cube_size[1] 

        self.PE = self._getValues()

    def _getValues(self,n=10000):
        PE = torch.zeros([self.W,self.H,self.D])
        for i in range(self.W):
            for j in range(self.H):
                for k in range(self.D):
                    p = np.power(n,(2*k)/self.D)
                    if (i+j)%2 == 0:
                        PE[i,j,k] = np.sin( (i+j)/p )
                    else:
                        PE[i,j,k] = np.cos( (i+j)/p )
        return PE


    """ Forward """
    def forward(self):
        return self.PE

"""
    Ref: https://github.com/codertimo/BERT-pytorch/blob/master/bert_pytorch/model/embedding/bert.py
"""
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x): # [120,L,D]
        return self.pe[:, :x.size(1)]
        
