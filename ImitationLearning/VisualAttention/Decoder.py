import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from   torch.autograd import Variable as V
from   torch.utils.data import Dataset,DataLoader
from   torchvision import models, transforms, utils

""" Attention Module 1
    ------------------
        * Input: feature [batch,L,D]
                 hidden  [1,batch,H]
        * Output: alpha  [batch,L,1]

    Architecture:
        pi   = Softmax[ w.h(t-1) + b ]
        attn = Softmax[ sum( feature o pi ) ]
"""
class Atten1(nn.Module):
    """ Constructor """
    def __init__(self):
        super(Atten1, self).__init__()
        # Parameters
        self.D = 64 # cube_size[0]
        self.L = 90 # cube_size[1]*cube_size[2]
        self.R = 5760 # self.L*self.D
        self.H = 1024 # hidden_size
        self.M = 1024 # hidden_size
        self.sequence_len =  20
        self.batch_size   = 120

        # Declare layers
        self.w = nn.Linear(self.H,self.D,bias=True)
        
        # Initialization
        torch.nn.init.xavier_uniform_(self.w.weight)
        self.softmax1 = nn.Softmax(dim=2)
        self.softmax2 = nn.Softmax(dim=1)
    
    """ Forward """
    def forward(self,feature,hidden):
        # Category pertinence
        pi =    self.w(hidden) # [1,batch,H]*[H,D] = [1,batch,D]
        pi    .transpose_(0,1) # [1,batch,D] -> [batch,1,D]
        pi = self.softmax1(pi) # [batch,1,D]
        
        attn = feature * pi       # [batch,L,D]x[batch,1,D]  = [batch,L,D]
        attn = torch.sum(attn,2,keepdim=True) # [batch,L,D] -> [batch,L,1]

        return self.softmax2(attn) # [batch,L,1]


""" Attention Module 2
    ------------------
    Architecture:
        pi   = Softmax[ w2( w1.h(t-1) + b ) ]
        attn = Softmax[ sum( feature o pi ) ]
"""
class Atten2(nn.Module):
    """ Constructor """
    def __init__(self):
        super(Atten2, self).__init__()
        # Parameters
        self.D = 64 # cube_size[0]
        self.L = 90 # cube_size[1]*cube_size[2]
        self.R = 5760 # self.L*self.D
        self.H = 1024 # hidden_size
        self.M = 1024 # hidden_size
        self.sequence_len =  20
        self.batch_size   = 120

        # Declare layers
        self.w1 = nn.Linear(self.H,self.D,bias= True)
        self.w2 = nn.Linear(self.D,self.D,bias=False)
        
        # Initialization
        torch.nn.init.xavier_uniform_(self.w.weight)
        self.softmax1 = nn.Softmax(dim=2)
        self.softmax2 = nn.Softmax(dim=1)
    
    """ Forward """
    def forward(self,feature,hidden):
        # Category pertinence
        pi =   self.w1(hidden) # [1,batch,H]*[H,D] = [1,batch,D]
        pi =   self.w2(hidden) # [1,batch,D]*[D,D] = [1,batch,D]
        pi    .transpose_(0,1) # [1,batch,D] -> [batch,1,D]
        pi = self.softmax1(pi) # [batch,1,D]
        
        attn = feature * pi       # [batch,L,D]x[batch,1,D]  = [batch,L,D]
        attn = torch.sum(attn,2,keepdim=True) # [batch,L,D] -> [batch,L,1]

        return self.softmax2(attn) # [batch,L,1]


""" Attention Module 3
    ------------------
    Architecture:
        pi   = Softmax[ w.h(t-1) + b ]
        attn = Softmax[ sum( feature o pi ) ]
"""
class Atten3(nn.Module):
    """ Constructor """
    def __init__(self):
        super(Atten3, self).__init__()
        # Parameters
        self.D = 64 # cube_size[0]
        self.L = 90 # cube_size[1]*cube_size[2]
        self.R = 5760 # self.L*self.D
        self.H = 1024 # hidden_size
        self.M = 1024 # hidden_size
        self.sequence_len =  20
        self.batch_size   = 120

        # Declare layers
        self.w = nn.Linear(self.H,self.D,bias=True)
        
        # Initialization
        torch.nn.init.xavier_uniform_(self.w.weight)
        self.softmax1 = nn.Softmax(dim=2)
        self.softmax2 = nn.Softmax(dim=1)
    
    """ Forward """
    def forward(self,feature,hidden):
        # Category pertinence
        pi =    self.w(hidden) # [1,batch,H]*[H,D] = [1,batch,D]
        pi    .transpose_(0,1) # [1,batch,D] -> [batch,1,D]
        pi = self.softmax1(pi) # [batch,1,D]
        
        attn = feature * pi        # [batch,L,D]x[batch,1,D]  = [batch,L,D]
        attn = torch.mean(attn,2,keepdim=True) # [batch,L,D] -> [batch,L,1]

        return self.softmax2(attn) # [batch,L,1]


""" Attention Module 4
    ------------------
    Architecture:
        pi   = Softmax[ w2( w1.h(t-1) + b ) ]
        attn = Softmax[ sum( feature o pi ) ]
"""
class Atten4(nn.Module):
    """ Constructor """
    def __init__(self):
        super(Atten4, self).__init__()
        # Parameters
        self.D = 64 # cube_size[0]
        self.L = 90 # cube_size[1]*cube_size[2]
        self.R = 5760 # self.L*self.D
        self.H = 1024 # hidden_size
        self.M = 1024 # hidden_size
        self.sequence_len =  20
        self.batch_size   = 120

        # Declare layers
        self.w1 = nn.Linear(self.H,self.D,bias= True)
        self.w2 = nn.Linear(self.D,self.D,bias=False)
        
        # Initialization
        torch.nn.init.xavier_uniform_(self.w.weight)
        self.softmax1 = nn.Softmax(dim=2)
        self.softmax2 = nn.Softmax(dim=1)
    
    """ Forward """
    def forward(self,feature,hidden):
        # Category pertinence
        pi =   self.w1(hidden) # [1,batch,H]*[H,D] = [1,batch,D]
        pi =   self.w2(hidden) # [1,batch,D]*[D,D] = [1,batch,D]
        pi    .transpose_(0,1) # [1,batch,D] -> [batch,1,D]
        pi = self.softmax1(pi) # [batch,1,D]
        
        attn = feature * pi        # [batch,L,D]x[batch,1,D]  = [batch,L,D]
        attn = torch.mean(attn,2,keepdim=True) # [batch,L,D] -> [batch,L,1]

        return self.softmax2(attn) # [batch,L,1]

    
""" [Kim2017] Decoder Module
    ------------------------
    Ref: Kim, J., & Canny, J. (2017). "Interpretable learning for self-driving 
         cars by visualizing causal attention". In Proceedings of the IEEE 
         international conference on computer vision (pp. 2942-2950).
    
        * Input: feature [batch,L,D]
                 hidden  [1,batch,H]
        * Output: alpha  [batch,L,1]
"""
class Kim2017(nn.Module):
    """ Constructor """
    def __init__(self,n_out):
        super(Kim2017, self).__init__()
        
        # Parameters
        self.D = 64 # cube_size[0]
        self.L = 90 # cube_size[1]*cube_size[2]
        self.R = 5760 # self.L*self.D
        self.H = 1024 # hidden_size
        self.M = 1024 # hidden_size
        self.sequence_len =  20
        self.batch_size   = 120
        self.n_out        = n_out
        
        # Declare layers
        self.attn = Attention()
        self.lstm = nn.LSTM( input_size = self.R,
                             hidden_size = self.H,
                             num_layers =      1)
        
        self.Wh = nn.Linear(self.H,self.M,bias=True )
        self.Wy = nn.Linear(self.R,self.M,bias=False)
        self.Wu = nn.Linear(self.M, n_out,bias=True )
        
        self.init_Wh   = nn.Linear(self.D,self.H,bias=True )
        self.init_Wc   = nn.Linear(self.D,self.H,bias=True )
        self.init_tanh = nn.Tanh()

        # Initialization
        torch.nn.init.xavier_uniform_(self.     Wh.weight)
        torch.nn.init.xavier_uniform_(self.     Wy.weight)
        torch.nn.init.xavier_uniform_(self.     Wu.weight)
        torch.nn.init.xavier_uniform_(self.init_Wh.weight)
        torch.nn.init.xavier_uniform_(self.init_Wc.weight)
        
        
    """ Initialize LSTM: hidden state and cell state
        Ref: Xu, Kelvin, et al. "Show, attend and tell: Neural 
            image caption generation with visual attention." 
            International conference on machine learning. 2015.

            * Input:  feature [batch,L,D]
            * Output: hidden  [1,batch,H]
                      cell    [1,batch,H]
    """
    def initializeLSTM(self,feature):
        with torch.no_grad():
            # Mean features
            feature = torch.mean(feature,1) # [batch,L,D] -> [batch,D]
            
            hidden = self.init_Wh(feature)  # [batch,D]*[D,H] -> [batch,H]
            hidden = self.init_tanh(hidden) # [batch,H]
            hidden = hidden.unsqueeze(0)    # [1,batch,H]

            cell = self.init_Wc(feature)    # [batch,D]*[D,H] -> [batch,H]
            cell = self.init_tanh(cell)     # [batch,H]
            cell = cell.unsqueeze(0)        # [1,batch,H]

            # (h,c) ~ [num_layers, batch, hidden_size]
            return hidden.contiguous(),cell.contiguous()
        
    """ Forward """
    def forward(self,feature):
        #
        # Training mode
        # -------------
        if self.training:
            sequence_len = self.sequence_len
            batch_size   = feature.shape[0]
            batch_size   = int(batch_size/sequence_len)
            out = torch.zeros([sequence_len,batch_size,self.n_out]).to( torch.device('cuda:0') )

            # Fragment
            sequence = feature.view( batch_size,sequence_len,self.L,self.D )
            sequence.transpose_(0,1)    # [sequence,batch,L,D]

            # Inicialize hidden state and cell state
            #   * hidden ~ [1,batch,H]
            #   * cell   ~ [1,batch,H]
            hidden,cell = self.initializeLSTM(sequence[0])

            # Sequence loop
            for k in range(sequence_len):
                # One time
                xt = sequence[k]        # [batch,L,D]
                
                # Visual Attention
                alpha = self.attn(xt,hidden)                # [batch,L,1]
                visual = xt * alpha                         # [batch,L,D]x[batch,L,1] = [batch,L,D]
                visual = visual.reshape(batch_size,self.R)  # [batch,R]
                visual = visual.unsqueeze(0)                # [1,batch,R]
                
                # LSTM
                #  * yt     ~ [sequence,batch,H]
                #  * hidden ~ [ layers ,batch,H]
                #  * cell   ~ [ layers ,batch,H]
                yt,(hidden,cell) = self.lstm(visual,(hidden,cell))
                
                # Control
                ut = self.Wh(hidden) + self.Wy(visual)               # [1,batch,M]
                ut = F.dropout(ut, p=0.5, training=self.training)
                ut = self.Wu(ut)        # [1,batch,M]*[M,n_outs] = [1,batch,n_outs]
                
                # Save sequence out
                out[k] = ut
            
            return out.transpose(0,1).contiguous().view(batch_size*sequence_len,self.n_out)
            
            
        #
        # Evaluation mode
        # ---------------
        else:
            # Fragment
            sequence   = feature # [batch,L,D]
            batch_size = feature.shape[0]

            # Inicialize hidden state and cell state
            #   * hidden ~ [1,batch,H]
            #   * cell   ~ [1,batch,H]
            hidden,cell = self.initializeLSTM(feature[0].unsqueeze(0))
            
            # Prediction container
            pred = torch.zeros([batch_size,self.n_out]).to( torch.device('cuda:0') )

            # Sequence loop
            for k in range(batch_size):
                # One time
                xt = sequence[k].unsqueeze(0)      # [1,L,D]

                # Visual Attention
                alpha = self.attn(xt,hidden)       # [batch,L,1]
                visual = xt * alpha                # [batch,L,D]x[batch,L,1] = [batch,L,D]
                visual = visual.reshape(1,self.R)  # [1,R]
                visual = visual.unsqueeze(0)       # [1,1,R]

                # LSTM
                #  * yt     ~ [sequence,batch,H]
                #  * hidden ~ [ layers ,batch,H]
                #  * cell   ~ [ layers ,batch,H]
                yt,(hidden,cell) = self.lstm(visual,(hidden,cell))

                # Control
                ut = self.Wh(hidden) + self.Wy(visual)           # [1,batch,M]
                ut = self.Wu(ut)        # [1,batch,M]*[M,n_outs] = [1,batch,n_outs]
                
                # Save sequence out
                pred[k] = ut

            return pred
        