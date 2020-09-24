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

class _Branch2(nn.Module):
    def __init__(self,n_out):
        super(_Branch2, self).__init__()
        # Declare layers
        self.fully1  = nn.Linear(1024,  256  )
        self.fully2s = nn.Linear( 256,   64  )
        self.fully3s = nn.Linear(  64,   1   )
        self.fully2v = nn.Linear( 256,   64  )
        self.fully3v = nn.Linear(  64,n_out-1)

        self.Sigmoid = nn.Sigmoid()

        # Initialization
        torch.nn.init.xavier_uniform_(self.fully1.weight)
        torch.nn.init.xavier_uniform_(self.fully2s.weight)
        torch.nn.init.xavier_uniform_(self.fully3s.weight)
        torch.nn.init.xavier_uniform_(self.fully2v.weight)
        torch.nn.init.xavier_uniform_(self.fully3v.weight)

    def forward(self,sig):
        hd = F.relu(self.fully1(sig))
        hd = F.dropout(hd, p=0.5, training=self.training)

        # Steering controller
        hs    = F.relu(self.fully2s( hd))
        steer = self.fully3s(hs)

        # Velocity controller
        hv  = F.relu(self.fully2v( hd))
        vel = self.Sigmoid(self.fully3v(hv))

        return torch.cat([steer,vel],dim=1)


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

        return y_pred
        
