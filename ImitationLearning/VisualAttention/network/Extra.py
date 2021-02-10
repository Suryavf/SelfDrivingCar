import torch
import torch.nn as nn

class Gating(nn.Module):
    def __init__(self, d_input, bg=0.1):
        super(Gating, self).__init__()
        self.Wr = nn.Linear(d_input, d_input)
        self.Ur = nn.Linear(d_input, d_input)
        self.Wz = nn.Linear(d_input, d_input)
        self.Uz = nn.Linear(d_input, d_input)
        self.Wg = nn.Linear(d_input, d_input)
        self.Ug = nn.Linear(d_input, d_input)
        self.bg = bg

        self.sigmoid = nn.Sigmoid()
        self.tanh    = nn.Tanh()

    def forward(self, x, y):
        r = self.sigmoid(self.Wr(y) + self.Ur(x))
        z = self.sigmoid(self.Wz(y) + self.Uz(x) - self.bg)
        h = self.tanh(self.Wg(y) + self.Ug(torch.mul(r, x)))

        g = (1-z)*x + z*h
        return g
        
