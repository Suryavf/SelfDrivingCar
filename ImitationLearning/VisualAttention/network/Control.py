import torch
import torch.nn as nn
import torch.nn.functional as F

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
    def __init__(self, cube_size, n_hidden):
        super(SumHiddenFeature, self).__init__()
        # Parameters
        self.D = cube_size[2]               #   64 # cube_size[0]
        self.L = cube_size[0]*cube_size[1]  #   90 # cube_size[1]*cube_size[2]
        self.R = self.L*self.D              # 5760 # self.L*self.D
        self.H = n_hidden                   #  512 # hidden_size
        self.M = n_hidden                   #  512 # hidden_size
        self.n_out = 3

        # Declare layers
        self.Wh = nn.Linear(self.H,self.M    ,bias=True )
        self.Wy = nn.Linear(self.R,self.M    ,bias=False)
        self.Wu = nn.Linear(self.M,self.n_out,bias=True )

        # Initialization
        torch.nn.init.xavier_uniform_(self.Wh.weight)
        torch.nn.init.xavier_uniform_(self.Wy.weight)
        torch.nn.init.xavier_uniform_(self.Wu.weight)

    def forward(self,feature,hidden):
        ut = self.Wh(hidden) + self.Wy(feature)          # [1,batch,M]
        ut = F.dropout(ut, p=0.5, training=self.training)
        ut = self.Wu(ut)        # [1,batch,M]*[M,n_outs] = [1,batch,n_outs]
        return ut
                

class _Branch(nn.Module):
    def __init__(self,n_out):
        super(_Branch, self).__init__()
        # Declare layers
        self.fully1 = nn.Linear(1024, 256 )
        self.fully2 = nn.Linear( 256,  64 )
        self.fully3 = nn.Linear(  64,n_out)

        # Initialization
        torch.nn.init.xavier_uniform_(self.fully1.weight)
        torch.nn.init.xavier_uniform_(self.fully2.weight)
        torch.nn.init.xavier_uniform_(self.fully3.weight)

    def forward(self,sig):
        h1 = F.relu(self.fully1(sig))
        h1 = F.dropout(h1, p=0.5, training=self.training)
        h2 = F.relu(self.fully2( h1))
        out = self.fully3(h2)
        return out


class BranchesModule(nn.Module):
    def __init__(self, cube_size, n_hidden, n_out=3):
        super(BranchesModule, self).__init__()
        # Parameters
        self.D     = cube_size[2]               # cube_size[0]
        self.L     = cube_size[0]*cube_size[1]  # cube_size[1]*cube_size[2]
        self.R     = self.L*self.D              # self.L*self.D
        self.H     = n_hidden                   # hidden_size
        self.M     = n_hidden                   # hidden_size
        self.n_out = n_out

        # Declare layers
        self.Wh = nn.Linear(self.H,self.M    ,bias=True )
        self.Wy = nn.Linear(self.R,self.M    ,bias=False)

        self.branches = nn.ModuleList([ _Branch(self.n_out) for i in range(4) ])

        # Initialization
        torch.nn.init.xavier_uniform_(self.Wh.weight)
        torch.nn.init.xavier_uniform_(self.Wy.weight)

    def forward(self,feature,hidden,mask):
        ut = self.Wh(hidden) + self.Wy(feature)          # [1,batch,M]
        ut = F.dropout(ut, p=0.1, training=self.training)

        # Branches
        y_pred = torch.cat( [out(ut) for out in self.branches], dim=1)
        y_pred = y_pred*mask
        #y_pred = y_pred.view(-1,3,4).sum(-1)

        return y_pred
        
