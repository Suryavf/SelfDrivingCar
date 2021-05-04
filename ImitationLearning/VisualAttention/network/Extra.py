import torch
import torch.nn as nn

class Gating(nn.Module):
    def __init__(self, d_input, bg=0.1):
        super(Gating, self).__init__()
        self.Wr = nn.Linear(2*d_input, d_input, bias=False)
        self.Wz = nn.Linear(2*d_input, d_input, bias=False)
        self.Wh = nn.Linear(2*d_input, d_input, bias=False)
        self.bg = bg

        self.Sigmoid = nn.Sigmoid()
        self.Tanh    = nn.Tanh()

    """ Forward 
          - x,y [batch,L,D]
    """
    def forward(self, x, y):
        # Concatenate
        xy = torch.cat([x,y],dim=1)

        r = self.Sigmoid( self.Wr(xy)           )
        z = self.Sigmoid( self.Wz(xy) - self.bg )

        h = torch.cat([y,r*x],dim=1)
        h = self.Tanh( h )
        g = (1-z)*x + z*h
        return g
        
