import torch
import torch.nn as nn
import torch.nn.functional as F

class SpeedRegModule(nn.Module):
    """ Constructor """
    def __init__(self,n_hidden):
        super(SpeedRegModule, self).__init__()
        # Parameters
        self.n_hidden = n_hidden

        self.w1 = nn.Linear(n_hidden,256)
        self.w2 = nn.Linear(   256  , 64)
        self.w3 = nn.Linear(    64  ,  1)

        # Initialization
        nn.init.xavier_uniform_(self.w1.weight)
        nn.init.xavier_uniform_(self.w2.weight)
        nn.init.xavier_uniform_(self.w3.weight)

    def forward(self,hidden):
        h = self.w1(hidden)
        h = F.relu(h)
        h = F.dropout(h, p=0.5, training=self.training)

        # -> CONV/FC -> BatchNorm -> ReLu(or other activation) -> Dropout -> CONV/FC ->
        # Ref: https://stackoverflow.com/questions/39691902/ordering-of-batch-normalization-and-dropout
        h = self.w2(h)
        h = F.relu(h)
        h = F.dropout(h, p=0.1, training=self.training)
        
        return self.w3(h)
        
