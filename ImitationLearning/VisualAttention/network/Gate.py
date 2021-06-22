import torch
import torch.nn as nn

class GRUGate(nn.Module):
    def __init__(self, d_input, bg=0.1):
        super(GRUGate, self).__init__()
        self.Wr = nn.Conv2d(d_input, d_input, kernel_size = 1, bias=False, stride=1)
        self.Ur = nn.Conv2d(d_input, d_input, kernel_size = 1, bias=False, stride=1)
        self.Wz = nn.Conv2d(d_input, d_input, kernel_size = 1, bias=False, stride=1)
        self.Uz = nn.Conv2d(d_input, d_input, kernel_size = 1, bias=False, stride=1)
        self.Wg = nn.Conv2d(d_input, d_input, kernel_size = 1, bias=False, stride=1)
        self.Ug = nn.Conv2d(d_input, d_input, kernel_size = 1, bias=False, stride=1)
        self.bg = bg

        self.sigmoid = nn.Sigmoid()
        self.tanh    = nn.Tanh()

        # Initialization
        # self.Wz.bias.data.fill_(-2)
        nn.init.xavier_uniform_(self.Wr.weight)
        nn.init.xavier_uniform_(self.Ur.weight)
        nn.init.xavier_uniform_(self.Wz.weight)
        nn.init.xavier_uniform_(self.Uz.weight)
        nn.init.xavier_uniform_(self.Wg.weight)
        nn.init.xavier_uniform_(self.Ug.weight)
        
    def forward(self, x, y):
        r = self.sigmoid(self.Wr(y) + self.Ur(x))
        z = self.sigmoid(self.Wz(y) + self.Uz(x) - self.bg)
        h = self.tanh(self.Wg(y) + self.Ug(torch.mul(r, x)))
        g = torch.mul(1 - z, x) + torch.mul(z, h)
        return g


class Gate(nn.Module):
    def __init__(self, d_input, bg=0.1):
        super(Gate, self).__init__()
        self.Wz = nn.Conv2d(d_input, d_input, kernel_size = 1, bias=False, stride=1)
        self.Uz = nn.Conv2d(d_input, d_input, kernel_size = 1, bias=False, stride=1)
        self.bg = bg

        self.sigmoid = nn.Sigmoid()
        self.tanh    = nn.Tanh()

        # Initialization
        # self.Wz.bias.data.fill_(-2)
        nn.init.xavier_uniform_(self.Wz.weight)
        nn.init.xavier_uniform_(self.Uz.weight)

    def init_bias(self):
        with torch.no_grad():
            self.Wz.bias.fill_(-2)  # Manually setting this bias to allow starting with markov process
            # Note -2 is the setting used in the paper stable transformers

    def forward(self, x, y):
        z = self.sigmoid(self.Wz(y) + self.Uz(x) - self.bg)
        g = torch.mul(1 - z, x) + torch.mul(z, y)
        return g    
        
