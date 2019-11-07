import torch
import torch.nn as nn

""" Sum Hidden Feature
    ------------------
        * Input: feature [1,batch,   R  ]
                 hidden  [1,batch,   H  ]
        * Output:    ut  [1,batch,n_outs]

    Architecture:
        ut = wu.[ wh.h(t-1) + wy.x(t) + b1 ] + b2
"""
class SumHiddenFeature(nn.Module):
    """ Constructor """
    def __init__(self):
        super(SumHiddenFeature, self).__init__()
        # Parameters
        self.D = 64 # cube_size[0]
        self.L = 90 # cube_size[1]*cube_size[2]
        self.R = 5760 # self.L*self.D
        self.H = 1024 # hidden_size
        self.M = 1024 # hidden_size
        self.n_out = 3

        # Declare layers
        self.Wh = nn.Linear(self.H,self.M    ,bias=True )
        self.Wy = nn.Linear(self.R,self.M    ,bias=False)
        self.Wu = nn.Linear(self.M,self.n_out,bias=False)

        # Initialization
        torch.nn.init.xavier_uniform_(self.Wh.weight)
        torch.nn.init.xavier_uniform_(self.Wy.weight)
        torch.nn.init.xavier_uniform_(self.Wu.weight)

    def forward(self,feature,hidden):
        ut = self.Wh(hidden) + self.Wy(feature)          # [1,batch,M]
        ut = self.Wu(ut)        # [1,batch,M]*[M,n_outs] = [1,batch,n_outs]
        return ut
        
