import torch
import torch.nn as nn

class CommandNet(nn.Module):
    """ Constructor """
    def __init__(self,n_encode=16):
        super(CommandNet, self).__init__()
        self.Wc   = nn.Linear( 4, n_encode, bias= True)
        self.ReLU = nn.ReLU()
        # Initialization
        nn.init.xavier_uniform_(self.Wc.weight)

    def forward(self,control):
        c = control*2-1
        c = self.Wc(c)
        return self.ReLU(c)
        

class CommandVelocityNet(nn.Module):
    """ Constructor """
    def __init__(self,n_encode=16):
        super(CommandVelocityNet, self).__init__()
        self.Wc   = nn.Linear(     4    , n_encode, bias=True)
        self.Wv   = nn.Linear(     1    , n_encode, bias=True)
        self.Wp   = nn.Linear(2*n_encode, n_encode, bias=True)
        self.ReLU = nn.ReLU()
        # Initialization
        nn.init.xavier_uniform_(self.Wc.weight)
        nn.init.xavier_uniform_(self.Wv.weight)
        nn.init.xavier_uniform_(self.Wp.weight)

    def forward(self,control,velocity):
        c = control*2-1
        c = self.ReLU( self.Wc(c) )
        v = self.ReLU( self.Wv(velocity) )

        p = torch.cat([c,v],dim=1)
        p = self.Wp(p)
        return self.ReLU(p)
        
