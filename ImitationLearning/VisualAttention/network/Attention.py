import torch
import torch.nn as nn

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
    def __init__(self,cube_size, n_hidden):
        super(Atten1, self).__init__()
        # Parameters
        self.D = cube_size[2]               #   64 # cube_size[0]
        self.L = cube_size[0]*cube_size[1]  #   90 # cube_size[1]*cube_size[2]
        self.R = self.L*self.D              # 5760 # self.L*self.D
        self.H = n_hidden                   #  512 # hidden_size
        self.M = n_hidden                   #  512 # hidden_size

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
    def __init__(self,cube_size, n_hidden):
        super(Atten2, self).__init__()
        # Parameters
        self.D = cube_size[2]               #   64 # cube_size[0]
        self.L = cube_size[0]*cube_size[1]  #   90 # cube_size[1]*cube_size[2]
        self.R = self.L*self.D              # 5760 # self.L*self.D
        self.H = n_hidden                   #  512 # hidden_size
        self.M = n_hidden                   #  512 # hidden_size

        # Declare layers
        self.w1 = nn.Linear(self.H,self.D,bias= True)
        self.w2 = nn.Linear(self.D,self.D,bias=False)
        self.sigmoid = nn.Sigmoid()

        # Initialization
        torch.nn.init.xavier_uniform_(self.w1.weight)
        torch.nn.init.xavier_uniform_(self.w2.weight)
        self.softmax1 = nn.Softmax(dim=2)
        self.softmax2 = nn.Softmax(dim=1)
    
    """ Forward """
    def forward(self,feature,hidden):
        # Category pertinence
        pi =   self.w1(hidden) # [1,batch,H]*[H,D] = [1,batch,D]
        pi =  self.sigmoid(pi)
        pi =   self.w2(  pi  ) # [1,batch,D]*[D,D] = [1,batch,D]
        pi    .transpose_(0,1) # [1,batch,D] -> [batch,1,D]
        
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
    def __init__(self,cube_size, n_hidden):
        super(Atten3, self).__init__()
        # Parameters
        self.D = cube_size[2]               #   64 # cube_size[0]
        self.L = cube_size[0]*cube_size[1]  #   90 # cube_size[1]*cube_size[2]
        self.R = self.L*self.D              # 5760 # self.L*self.D
        self.H = n_hidden                   #  512 # hidden_size
        self.M = n_hidden                   #  512 # hidden_size
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
    def __init__(self, cube_size, n_hidden):
        super(Atten4, self).__init__()
        # Parameters
        self.D = cube_size[2]               #   64 # cube_size[0]
        self.L = cube_size[0]*cube_size[1]  #   90 # cube_size[1]*cube_size[2]
        self.R = self.L*self.D              # 5760 # self.L*self.D
        self.H = n_hidden                   #  512 # hidden_size
        self.M = n_hidden                   #  512 # hidden_size

        # Declare layers
        self.w1 = nn.Linear(self.H,self.D,bias= True)
        self.w2 = nn.Linear(self.D,self.D,bias=False)
        self.sigmoid = nn.Sigmoid()
        
        # Initialization
        torch.nn.init.xavier_uniform_(self.w1.weight)
        torch.nn.init.xavier_uniform_(self.w2.weight)
        self.softmax2 = nn.Softmax(dim=1)
    
    """ Forward """
    def forward(self,feature,hidden):
        # Category pertinence
        pi =  self. w1(hidden) # [1,batch,H]*[H,D] = [1,batch,D]
        pi =  self.sigmoid(pi)
        pi =  self. w2(  pi  ) # [1,batch,D]*[D,D] = [1,batch,D]
        pi    .transpose_(0,1) # [1,batch,D] -> [batch,1,D]
        
        attn = feature * pi        # [batch,L,D]x[batch,1,D]  = [batch,L,D]
        attn = torch.mean(attn,2,keepdim=True) # [batch,L,D] -> [batch,L,1]

        return self.softmax2(attn) # [batch,L,1] 
        

""" Attention Module 5
    ------------------
    Architecture:
        
"""
class Atten5(nn.Module):
    """ Constructor """
    def __init__(self, cube_size, n_hidden):
        super(Atten5, self).__init__()
        # Parameters
        self.D = cube_size[2]               #  depth
        self.L = cube_size[0]*cube_size[1]  #  h x w
        self.R = self.L*self.D              #  L x D
        self.H = n_hidden                   #  hidden_size
        self.M = n_hidden                   #  hidden_size

        # Filtering 
        self.filteringLSTM = nn.LSTM( input_size = self.H, hidden_size = 512)
        self.wf1 = nn.Linear(512,   256,bias=True)
        self.wf2 = nn.Linear(256,self.L,bias=False)
        
        # Pigeonholing 
        self.pigeonholingLSTM = nn.LSTM(input_size = self.H, hidden_size = 512)
        self.wp1 = nn.Linear(512,   256,bias=True)
        self.wp2 = nn.Linear(256,self.D,bias=False)

        # Initialization
        self.   filteringLSTM.reset_parameters()
        self.pigeonholingLSTM.reset_parameters()
        torch.nn.init.xavier_uniform_(self.wf1.weight)
        torch.nn.init.xavier_uniform_(self.wf2.weight)
        torch.nn.init.xavier_uniform_(self.wp1.weight)
        torch.nn.init.xavier_uniform_(self.wp2.weight)

        self.ReLu    = nn.ReLU()
        self.Softmax = nn.Softmax(1)

    """ Forward """
    def forward(self,feature,hidden):
        # Filtering
        _,(hf,_) = self.filteringLSTM(hidden)
        xf = self.ReLu( self.wf1(hf) )  # [1,batch,a]*[a,b] = [1,batch,b]
        xf = self.ReLu( self.wf2(xf) )  # [1,batch,b]*[b,L] = [1,batch,L]
        xf = xf.squeeze(0)              # [1,batch,L] -> [batch,L]    
        
        # Pigeonholing
        _,(hp,_) = self.pigeonholingLSTM(hidden)
        xp = self.ReLu( self.wp1(hp) )  # [1,batch,a]*[a,b] = [1,batch,b]
        xp = self.ReLu( self.wp2(xp) )  # [1,batch,c]*[c,D] = [1,batch,D]
        xp = xp.squeeze(0)              # [1,batch,D] -> [batch,D]    

        # Attention maps
        alpha = self.Softmax( feature.mean(2)*xf )    # [batch,L]
        beta  = self.Softmax( feature.mean(1)*xp )    # [batch,D]
        return alpha.unsqueeze(2), beta.unsqueeze(1)
        

""" Attention Module 6
    ------------------
    Architecture:
        
"""
class Atten6(nn.Module):
    """ Constructor """
    def __init__(self, cube_size, n_hidden):
        super(Atten6, self).__init__()
        # Parameters
        self.D = cube_size[2]               #  depth
        self.L = cube_size[0]*cube_size[1]  #  h x w
        self.R = self.L*self.D              #  L x D
        self.H = n_hidden                   #  hidden_size
        self.M = n_hidden                   #  hidden_size

        # Filtering 
        self.filteringLSTM = nn.LSTM( input_size = self.H, hidden_size = 512)
        self.wfL = nn.Linear(1024,  256,bias=True)
        self.wfR = nn.Linear( 512,  256,bias=False)

        self.wf = nn.Linear(256,self.L,bias=True)
        self.avgFil = nn.AdaptiveAvgPool1d(1)
        
        # Pigeonholing 
        self.pigeonholingLSTM = nn.LSTM(input_size = self.H, hidden_size = 512)
        self.wpL = nn.Linear(1024,  256,bias=True)
        self.wpR = nn.Linear( 512,  256,bias=False)
        
        self.wp = nn.Linear(256,self.D,bias=True)
        self.avgPig = nn.AdaptiveAvgPool1d(1)

        # Initialization
        self.   filteringLSTM.reset_parameters()
        self.pigeonholingLSTM.reset_parameters()
        torch.nn.init.xavier_uniform_(self.wfL.weight)
        torch.nn.init.xavier_uniform_(self.wfR.weight)
        torch.nn.init.xavier_uniform_(self.wf .weight)
        torch.nn.init.xavier_uniform_(self.wpL.weight)
        torch.nn.init.xavier_uniform_(self.wpR.weight)
        torch.nn.init.xavier_uniform_(self.wp .weight)

        self.ReLu    = nn.ReLU()
        self.Softmax = nn.Softmax(1)

    """ Forward """
    def forward(self,feature,hidden):
        # Filtering
        _,(hf,_) = self.filteringLSTM(hidden)

        xfr = self.ReLu( self.wfR(hf) )     # [1,batch,a]*[a,b] = [1,batch,b]
        xfl = self.ReLu( self.wfL(hidden) ) # [1,batch,a]*[a,b] = [1,batch,b]
        
        xf = self.ReLu( self.wf( xfl+xfr ) )    # [1,batch,b]*[b,L] = [1,batch,L]
        xf = xf.squeeze(0)                      # [1,batch,L] -> [batch,L]    
        
        # Pigeonholing
        _,(hp,_) = self.pigeonholingLSTM(hidden)

        xpr = self.ReLu( self.wpR(hp) )     # [1,batch,a]*[a,b] = [1,batch,b]
        xpl = self.ReLu( self.wpL(hidden) ) # [1,batch,a]*[a,b] = [1,batch,b]

        xp = self.ReLu( self.wp( xpl+xpr ) )    # [1,batch,c]*[c,D] = [1,batch,D]
        xp = xp.squeeze(0)                      # [1,batch,D] -> [batch,D]    

        # Attention maps
        featureFil = self.avgFil(feature               ).squeeze(2)
        featurePig = self.avgPig(feature.transpose(1,2)).squeeze(2)
        alpha = self.Softmax( featureFil*xf )    # [batch,L]
        beta  = self.Softmax( featurePig*xp )    # [batch,D]
        
        return alpha.unsqueeze(2), beta.unsqueeze(1)
        

""" Attention Module 7
    ------------------
    Architecture:
        
"""
class Atten7(nn.Module):
    """ Constructor """
    def __init__(self, cube_size, n_hidden):
        super(Atten7, self).__init__()
        # Parameters
        self.D = cube_size[2]               #  depth
        self.L = cube_size[0]*cube_size[1]  #  h x w
        self.R = self.L*self.D              #  L x D
        self.H = n_hidden                   #  hidden_size
        self.M = n_hidden                   #  hidden_size

        # Filtering 
        self.filteringLSTM = nn.LSTM( input_size = self.H, hidden_size = 512)
        self.wfLh = nn.Linear(1024,  256,bias=True)
        self.wfRh = nn.Linear( 512,  256,bias=False)

        self.wfh = nn.Linear(  256 ,self.L,bias=True)
        self.wfx = nn.Linear(self.L,   1  ,bias=True)
        
        # Pigeonholing 
        self.pigeonholingLSTM = nn.LSTM(input_size = self.H, hidden_size = 512)
        self.wpLh = nn.Linear(1024,  256,bias=True)
        self.wpRh = nn.Linear( 512,  256,bias=False)
        
        self.wph = nn.Linear(  256 ,self.D,bias=True)
        self.wpx = nn.Linear(self.D,   1  ,bias=True)

        # Initialization
        self.   filteringLSTM.reset_parameters()
        self.pigeonholingLSTM.reset_parameters()

        torch.nn.init.xavier_uniform_(self.wfLh.weight)
        torch.nn.init.xavier_uniform_(self.wfRh.weight)
        torch.nn.init.xavier_uniform_(self. wfh.weight)
        torch.nn.init.xavier_uniform_(self. wfx.weight)

        torch.nn.init.xavier_uniform_(self.wpLh.weight)
        torch.nn.init.xavier_uniform_(self.wpRh.weight)
        torch.nn.init.xavier_uniform_(self. wph.weight)
        torch.nn.init.xavier_uniform_(self. wpx.weight)

        self.ReLu    = nn.ReLU()
        self.Softmax = nn.Softmax(1)

    """ Forward """
    def forward(self,feature,hidden):
        # Filtering
        _,(hfilter,_) = self.filteringLSTM(hidden)

        vfRh = self.ReLu( self.wfRh(hfilter) )  # [1,batch,a]*[a,b] = [1,batch,b]
        vfLh = self.ReLu( self.wfLh(hidden ) )  # [1,batch,a]*[a,b] = [1,batch,b]
        
        vfh = self.ReLu( self.wfh( vfRh+vfLh ) )    # [1,batch,b]*[b,L] = [1,batch,L]
        vfh = vfh.squeeze(0)                        # [1,batch,L] -> [batch,L]    
        
        # Pigeonholing
        _,(hpigeon,_) = self.pigeonholingLSTM(hidden)

        vpRh = self.ReLu( self.wpRh(hpigeon) )  # [1,batch,a]*[a,b] = [1,batch,b]
        vpLh = self.ReLu( self.wpLh(hidden ) )  # [1,batch,a]*[a,b] = [1,batch,b]

        vph = self.ReLu( self.wph( vpRh+vpLh ) )    # [1,batch,c]*[c,D] = [1,batch,D]
        vph = vph.squeeze(0)                        # [1,batch,D] -> [batch,D]    

        # Feature reduction
        featureFil = self.ReLu( self.wfx(        feature       ) ).squeeze(2)
        featurePig = self.ReLu( self.wpx(feature.transpose(1,2)) ).squeeze(2)

        # Attention maps
        alpha = self.Softmax( featureFil*vfh )    # [batch,L]
        beta  = self.Softmax( featurePig*vph )    # [batch,D]
        
        return alpha.unsqueeze(2), beta.unsqueeze(1)
        
