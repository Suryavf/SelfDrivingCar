import torch
import torch.nn as nn

""" Convolutional Neuronal Network - 5 layers
    -----------------------------------------
    Ref: Kim, J., & Canny, J. (2017). "Interpretable learning for self-driving 
         cars by visualizing causal attention". In Proceedings of the IEEE 
         international conference on computer vision (pp. 2942-2950).
    
        * Input: img [batch,h,w]
        * Output: xt [batch,L,D]
"""
class CNN5(nn.Module):
    """ Constructor """
    def __init__(self):
        super(CNN5, self).__init__()
        
        # Layers
        self.conv1 = nn.Conv2d( 3, 24, kernel_size=5, stride=2, bias=False)
        self.conv2 = nn.Conv2d(24, 36, kernel_size=5, stride=2, bias=False)
        self.conv3 = nn.Conv2d(36, 48, kernel_size=3, stride=2, bias=False)
        self.conv4 = nn.Conv2d(48, 64, kernel_size=3, stride=1, bias=False)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1, bias=False)

        self.batchN1 = nn.BatchNorm2d(24)
        self.batchN2 = nn.BatchNorm2d(36)
        self.batchN3 = nn.BatchNorm2d(48)
        self.batchN4 = nn.BatchNorm2d(64)
        self.batchN5 = nn.BatchNorm2d(64)
        
        self.ReLU = nn.ReLU()
        
        # Initialize
        torch.nn.init.xavier_uniform_(self.conv1.weight)
        torch.nn.init.xavier_uniform_(self.conv2.weight)
        torch.nn.init.xavier_uniform_(self.conv3.weight)
        torch.nn.init.xavier_uniform_(self.conv4.weight)
        torch.nn.init.xavier_uniform_(self.conv5.weight)
        self.batchN1.reset_parameters()
        self.batchN2.reset_parameters()
        self.batchN3.reset_parameters()
        self.batchN4.reset_parameters()
        self.batchN5.reset_parameters()


    """ Forward """
    def forward(self,x):
        # Layer 1
        x = self.  conv1(x)
        x = self.batchN1(x)
        x = self.   ReLU(x)

        # Layer 2
        x = self.  conv2(x)
        x = self.batchN2(x)
        x = self.   ReLU(x)

        # Layer 3
        x = self.  conv3(x)
        x = self.batchN3(x)
        x = self.   ReLU(x)
        
        # Layer 4
        x = self.  conv4(x)
        x = self.batchN4(x)
        x = self.   ReLU(x)

        # Layer 5
        x = self.  conv5(x)
        x = self.batchN5(x)
        x = self.   ReLU(x)                     # [batch,D,h,w]
        
        x = x.flatten(start_dim=2, end_dim=3)   # [batch,D,L]
        return x.transpose(1, 2)                # [batch,L,D]
        
