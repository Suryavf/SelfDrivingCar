import math 
import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.avgFiltering = nn.AdaptiveAvgPool1d(1)
        
        # Pigeonholing 
        self.pigeonholingLSTM = nn.LSTM(input_size = self.H, hidden_size = 512)
        self.wpL = nn.Linear(1024,  256,bias=True)
        self.wpR = nn.Linear( 512,  256,bias=False)
        
        self.wp = nn.Linear(256,self.D,bias=True)
        self.avgPigeonholing = nn.AdaptiveAvgPool1d(1)

        # Initialization
        self.   filteringLSTM.reset_parameters()
        self.pigeonholingLSTM.reset_parameters()
        torch.nn.init.xavier_uniform_(self.wfL.weight)
        torch.nn.init.xavier_uniform_(self.wfR.weight)
        torch.nn.init.xavier_uniform_(self.wf .weight)
        torch.nn.init.xavier_uniform_(self.wpL.weight)
        torch.nn.init.xavier_uniform_(self.wpR.weight)
        torch.nn.init.xavier_uniform_(self.wp .weight)

        self.ReLu    = nn.LeakyReLU()
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
        featureFil = self.avgFiltering   (feature               ).squeeze(2)
        featurePig = self.avgPigeonholing(feature.transpose(1,2)).squeeze(2)
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
        self.wfLh = nn.Linear(1024,256,bias=True)
        self.wfRh = nn.Linear( 512,256,bias=False)

        self.wfh = nn.Linear(  256 ,self.L,bias=True)
        self.wfx = nn.Linear(self.D,   1  ,bias=True)
        
        # Pigeonholing 
        self.pigeonholingLSTM = nn.LSTM(input_size = self.H, hidden_size = 512)
        self.wpLh = nn.Linear(1024,256,bias=True)
        self.wpRh = nn.Linear( 512,256,bias=False)
        
        self.wph = nn.Linear(  256 ,self.D,bias=True)
        self.wpx = nn.Linear(self.L,   1  ,bias=True)

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
        

""" Attention Module 8
    ------------------
    Architecture:
        
"""
class Atten8(nn.Module):
    """ Constructor """
    def __init__(self, cube_size, n_hidden):
        super(Atten8, self).__init__()
        # Parameters
        self.D = cube_size[2]               #  depth
        self.L = cube_size[0]*cube_size[1]  #  h x w
        self.R = self.L*self.D              #  L x D
        self.H = n_hidden                   #  hidden_size
        self.M = n_hidden                   #  hidden_size

        # Filtering 
        self.filteringLSTM = nn.LSTM( input_size = self.H, hidden_size = 512)
        self.wfA = nn.Linear(1024,256,bias= True)
        self.wfR = nn.Linear( 512,256,bias=False)

        self.wfn    = nn.Linear(  256 ,self.L,bias=True)
        self.avgFiltering = nn.AdaptiveAvgPool1d(1)
        self.wf     = nn.Linear(self.L,self.L,bias=True)

        # Pigeonholing 
        self.pigeonholingLSTM = nn.LSTM(input_size = self.H, hidden_size = 512)
        self.wpA = nn.Linear(1024,256,bias= True)
        self.wpR = nn.Linear( 512,256,bias=False)
        
        self.wpn    = nn.Linear(  256 ,self.D,bias=True)
        self.avgPigeonholing = nn.AdaptiveAvgPool1d(1)
        self.wp     = nn.Linear(self.D,self.D,bias=True)

        # Initialization
        self.   filteringLSTM.reset_parameters()
        self.pigeonholingLSTM.reset_parameters()
        
        torch.nn.init.xavier_uniform_(self.wfA.weight)
        torch.nn.init.xavier_uniform_(self.wfR.weight)
        torch.nn.init.xavier_uniform_(self.wfn.weight)
        torch.nn.init.xavier_uniform_(self.wf .weight)
        
        torch.nn.init.xavier_uniform_(self.wpA.weight)
        torch.nn.init.xavier_uniform_(self.wpR.weight)
        torch.nn.init.xavier_uniform_(self.wpn.weight)
        torch.nn.init.xavier_uniform_(self.wp .weight)

        self.ReLu    = nn.ReLU()
        self.Softmax = nn.Softmax(1)

    """ Forward """
    def forward(self,feature,hidden):
        # Filtering
        _,(hf,_) = self.filteringLSTM(hidden)

        xfr = self.ReLu( self.wfR(hf) )     # [1,batch,a]*[a,b] = [1,batch,b]
        xfa = self.ReLu( self.wfA(hidden) ) # [1,batch,a]*[a,b] = [1,batch,b]
        
        xf = self.ReLu( self.wfn( xfa+xfr ) )   # [1,batch,b]*[b,L] = [1,batch,L]
        xf = xf.squeeze(0)                      # [1,batch,L] -> [batch,L]    
        
        # Pigeonholing
        _,(hp,_) = self.pigeonholingLSTM(hidden)

        xpr = self.ReLu( self.wpR(hp) )     # [1,batch,a]*[a,b] = [1,batch,b]
        xpa = self.ReLu( self.wpA(hidden) ) # [1,batch,a]*[a,b] = [1,batch,b]

        xp = self.ReLu( self.wpn( xpa+xpr ) )   # [1,batch,c]*[c,D] = [1,batch,D]
        xp = xp.squeeze(0)                      # [1,batch,D] -> [batch,D]    

        # Attention maps
        featureFil = self.avgFiltering   (feature               ).squeeze(2)
        featurePig = self.avgPigeonholing(feature.transpose(1,2)).squeeze(2)
        alpha = featureFil*xf   # [batch,L]
        beta  = featurePig*xp   # [batch,D]
        
        alpha = self.Softmax( self.wf(alpha) )
        beta  = self.Softmax( self.wp( beta) )

        return alpha.unsqueeze(2), beta.unsqueeze(1)
        

""" Attention Module 9
    ------------------
    Architecture:
        
"""
class Atten9(nn.Module):
    """ Constructor """
    def __init__(self, cube_size, n_hidden):
        super(Atten9, self).__init__()
        # Parameters
        self.D = cube_size[2]               #  depth
        self.L = cube_size[0]*cube_size[1]  #  h x w
        self.R = self.L*self.D              #  L x D
        self.H = n_hidden                   #  hidden_size
        self.M = n_hidden                   #  hidden_size

        self. inH = int(n_hidden/2) # 512 256
        self.medH = int(n_hidden/4) # 256 128

        # Filtering 
        self.filteringLSTM = nn.LSTM( input_size = self.H, hidden_size = self.inH)
        self.wfL = nn.Linear(self.  H,self.medH,bias=True)
        self.wfR = nn.Linear(self.inH,self.medH,bias=False)
        
        self.wf = nn.Linear(self.medH,self.L,bias=True)
        self.avgFiltering = nn.AdaptiveAvgPool1d(1)
        
        # Pigeonholing 
        self.pigeonholingLSTM = nn.LSTM(input_size = self.H, hidden_size = self.inH)
        self.wpL = nn.Linear(self.  H,self.medH,bias=True)
        self.wpR = nn.Linear(self.inH,self.medH,bias=False)
        
        self.wp = nn.Linear(self.medH,self.D,bias=True)
        self.avgPigeonholing = nn.AdaptiveAvgPool1d(1)
        
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
        self.Sigmoid = nn.Sigmoid()
        self.Softmax = nn.Softmax(1)


    def norm2(self,x):
        y = self.Sigmoid(x)**2
        y = x.mean(1) + 10**-6
        y = torch.sqrt(y)

        return x/y.view(x.shape[0],1)

    def norm4(self,x):
        y = self.Sigmoid(x)**4
        y = x.mean(1) + 10**-12
        y = torch.sqrt(y)
        y = torch.sqrt(y)

        return x/y.view(x.shape[0],1)


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

        xpr = self.ReLu( self.wpR(  hp  ) ) # [1,batch,a]*[a,b] = [1,batch,b]
        xpl = self.ReLu( self.wpL(hidden) ) # [1,batch,a]*[a,b] = [1,batch,b]

        xp = self.ReLu( self.wp( xpl+xpr ) )    # [1,batch,c]*[c,D] = [1,batch,D]
        xp = xp.squeeze(0)                      # [1,batch,D] -> [batch,D]    

        # Attention maps
        featureFil = self.avgFiltering   (feature               ).squeeze(2)
        featurePig = self.avgPigeonholing(feature.transpose(1,2)).squeeze(2)
        alpha = self.norm2( featureFil*xf )    # [batch,L]
        beta  = self.norm2( featurePig*xp )    # [batch,D]
        
        return alpha.unsqueeze(2), beta.unsqueeze(1)
        

""" Attention Module 10
    -------------------
    Architecture:
        
"""
class Atten10(nn.Module):
    """ Constructor """
    def __init__(self, cube_size, n_hidden):
        super(Atten10, self).__init__()
        # Parameters
        self.D = cube_size[2]               #  depth
        self.L = cube_size[0]*cube_size[1]  #  h x w
        self.R = self.L*self.D              #  L x D
        self.H = n_hidden                   #  hidden_size
        self.M = n_hidden                   #  hidden_size

        # Filtering 
        self.filteringLSTM = nn.LSTM( input_size = self.H, hidden_size = 512)
        self.wfL = nn.Linear(1024,self.L,bias=True )
        self.wfR = nn.Linear( 512,self.L,bias=False)

        self.avgFiltering = nn.AdaptiveAvgPool1d(1)
        self.wf = nn.Linear(self.L,self.L,bias=True)
        
        # Pigeonholing 
        self.pigeonholingLSTM = nn.LSTM(input_size = self.H, hidden_size = 512)
        self.wpL = nn.Linear(1024,self.D,bias=True )
        self.wpR = nn.Linear( 512,self.D,bias=False)

        self.avgPigeonholing = nn.AdaptiveAvgPool1d(1)
        self.wp = nn.Linear(self.D,self.D,bias=True)

        # Initialization
        self.   filteringLSTM.reset_parameters()
        self.pigeonholingLSTM.reset_parameters()
        torch.nn.init.xavier_uniform_(self.wfL.weight)
        torch.nn.init.xavier_uniform_(self.wfR.weight)
        torch.nn.init.xavier_uniform_(self.wpL.weight)
        torch.nn.init.xavier_uniform_(self.wpR.weight)
        torch.nn.init.xavier_uniform_(self.wf .weight)
        torch.nn.init.xavier_uniform_(self.wp .weight)

        self.ReLu    = nn.ReLU()
        self.Sigmoid = nn.Sigmoid()
        self.Softmax = nn.Softmax(1)


    def norm2(self,x):
        y = self.Sigmoid(x)**2
        y = y.mean(1) + 10**-6
        y = torch.sqrt(y)

        return x/y.view(x.shape[0],1)

    def norm4(self,x):
        y = self.Sigmoid(x)**4
        y = y.mean(1) + 10**-12
        y = torch.sqrt(y)
        y = torch.sqrt(y)

        return x/y.view(x.shape[0],1)


    """ Forward """
    def forward(self,feature,hidden):
        # Filtering
        _,(hf,_) = self.filteringLSTM(hidden)

        xfr = self.wfR(hf)      # [1,batch,a]*[a,b] = [1,batch,b]
        xfl = self.wfL(hidden)  # [1,batch,a]*[a,b] = [1,batch,b]
        
        xf = self.ReLu( xfl+xfr )   # [1,batch,b]*[b,L] = [1,batch,L]
        xf = xf.squeeze(0)          # [1,batch,L] -> [batch,L]    
        
        # Pigeonholing
        _,(hp,_) = self.pigeonholingLSTM(hidden)

        xpr = self.wpR(hp)      # [1,batch,a]*[a,b] = [1,batch,b]
        xpl = self.wpL(hidden)  # [1,batch,a]*[a,b] = [1,batch,b]
        
        xp = self.ReLu( xpl+xpr )    # [1,batch,c]*[c,D] = [1,batch,D]
        xp = xp.squeeze(0)                      # [1,batch,D] -> [batch,D]    

        # Attention maps
        featureFil = self.avgFiltering   (feature               ).squeeze(2)
        featureFil = self.Sigmoid( self.wf( featureFil ) )

        featurePig = self.avgPigeonholing(feature.transpose(1,2)).squeeze(2)
        featurePig = self.Sigmoid( self.wp( featurePig ) )

        alpha = self.norm4( featureFil*xf )    # [batch,L]
        beta  = self.norm4( featurePig*xp )    # [batch,D]
        
        return alpha.unsqueeze(2), beta.unsqueeze(1)
        

class Atten11(nn.Module):
    """ Constructor """
    def __init__(self, cube_size, n_hidden):
        super(Atten11, self).__init__()
        # Parameters
        self.D = cube_size[2]               #  depth
        self.L = cube_size[0]*cube_size[1]  #  h x w
        self.R = self.L*self.D              #  L x D
        self.H = n_hidden                   #  hidden_size
        self.M = n_hidden                   #  hidden_size

        # Filtering 
        self.filteringLSTM = nn.LSTM( input_size = self.H, hidden_size = 512)
        self.wfL = nn.Linear(1024,self.D,bias=True )
        self.wfR = nn.Linear( 512,self.D,bias=False)

        self.avgFiltering = nn.AdaptiveAvgPool1d(1)
        
        # Pigeonholing 
        self.pigeonholingLSTM = nn.LSTM(input_size = self.H, hidden_size = 512)
        self.wpL = nn.Linear(1024,self.D,bias=True )
        self.wpR = nn.Linear( 512,self.D,bias=False)
        self.avgPigeonholing = nn.AdaptiveAvgPool1d(1)
        
        self.wp = nn.Linear(self.D,self.D,bias=True)

        # Initialization
        self.   filteringLSTM.reset_parameters()
        self.pigeonholingLSTM.reset_parameters()
        torch.nn.init.xavier_uniform_(self.wfL.weight)
        torch.nn.init.xavier_uniform_(self.wfR.weight)
        torch.nn.init.xavier_uniform_(self.wpL.weight)
        torch.nn.init.xavier_uniform_(self.wpR.weight)
        torch.nn.init.xavier_uniform_(self.wp .weight)

        self.ReLu    = nn.ReLU()
        self.ReLu6   = nn.ReLU6()
        self.Sigmoid = nn.Sigmoid()
        self.Tanh    = nn.Tanh()
        self.Softmax = nn.Softmax(1)
        self.BNormF  = nn.BatchNorm1d(self.L)
        self.BNormP  = nn.BatchNorm1d(self.D)


    def norm2(self,x):
        y = self.Tanh(x)**2
        y = y.mean(1) + 10**-6
        y = torch.sqrt(y)

        return x/y.view(x.shape[0],1)

    def norm4(self,x):
        x = self.Tanh(x)
        y = x**4
        y = y.sum(1) + 10**-12
        y = torch.sqrt(y)
        y = torch.sqrt(y)

        return x/y.view(x.shape[0],1)


    """ Forward """
    def forward(self,feature,hidden):
        # Filtering
        _,(hf,_) = self.filteringLSTM(hidden)

        xfr = self.wfR(hf)      # [1,batch,a]*[a,b] = [1,batch,b]
        xfl = self.wfL(hidden)  # [1,batch,a]*[a,b] = [1,batch,b]
        
        xf = self.ReLu( xfl+xfr )   # [1,batch,b]*[b,L] = [1,batch,L]
        xf = xf.transpose(0,1)      # [1,batch,D] -> [batch,1,D] 

        # Pigeonholing
        _,(hp,_) = self.pigeonholingLSTM(hidden)

        xpr = self.wpR(hp)      # [1,batch,a]*[a,b] = [1,batch,b]
        xpl = self.wpL(hidden)  # [1,batch,a]*[a,b] = [1,batch,b]
        
        xp = self.ReLu( xpl+xpr )    # [1,batch,c]*[c,D] = [1,batch,D]
        xp = xp.squeeze(0)                      # [1,batch,D] -> [batch,D]    

        # Attention maps
        featureFil = self.avgFiltering ( feature * xf ).squeeze(2)
        alpha = self.norm4( featureFil )    # [batch,L]
        
        featurePig = self.avgPigeonholing(feature.transpose(1,2)).squeeze(2)
        featurePig = self.Sigmoid( self.wp( featurePig ) )
        beta  = self.norm4( featurePig*xp )    # [batch,D]
        
        return alpha.unsqueeze(2), beta.unsqueeze(1)
        

class Atten12(nn.Module):
    """ Constructor """
    def __init__(self, cube_size, n_hidden):
        super(Atten12, self).__init__()
        # Parameters
        self.D = cube_size[2]               #  depth
        self.L = cube_size[0]*cube_size[1]  #  h x w
        self.R = self.L*self.D              #  L x D
        self.H = n_hidden                   #  hidden_size
        self.M = n_hidden                   #  hidden_size

        # Filtering 
        self.filteringLSTM = nn.LSTM( input_size = self.H, hidden_size = 512)
        self.wfL = nn.Linear(1024,self.D,bias=True )
        self.wfR = nn.Linear( 512,self.D,bias=False)

        self.avgFiltering = nn.AdaptiveAvgPool1d(1)
        
        # Pigeonholing 
        self.pigeonholingLSTM = nn.LSTM(input_size = self.H, hidden_size = 512)
        self.wpL = nn.Linear(1024,self.D,bias=True )
        self.wpR = nn.Linear( 512,self.D,bias=False)
        self.avgPigeonholing = nn.AdaptiveAvgPool1d(1)
        
        self.wf = nn.Linear(self.D,self.D,bias=True)
        self.wp = nn.Linear(self.D,self.D,bias=True)

        # Initialization
        self.   filteringLSTM.reset_parameters()
        self.pigeonholingLSTM.reset_parameters()
        torch.nn.init.xavier_uniform_(self.wfL.weight)
        torch.nn.init.xavier_uniform_(self.wfR.weight)
        torch.nn.init.xavier_uniform_(self.wpL.weight)
        torch.nn.init.xavier_uniform_(self.wpR.weight)
        torch.nn.init.xavier_uniform_(self.wf .weight)
        torch.nn.init.xavier_uniform_(self.wp .weight)

        self.ReLu    = nn.ReLU()
        self.ReLu6   = nn.ReLU6()
        self.Sigmoid = nn.Sigmoid()
        self.Tanh    = nn.Tanh()
        self.Softmax = nn.Softmax(1)
        self.BNormF  = nn.BatchNorm1d(self.L)
        self.BNormP  = nn.BatchNorm1d(self.D)


    def norm2(self,x):
        y = self.Tanh(x)**2
        y = y.mean(1) + 10**-6
        y = torch.sqrt(y)

        return x/y.view(x.shape[0],1)

    def norm4(self,x):
        x = self.Tanh(x)
        y = x**4
        y = y.sum(1) + 10**-12
        y = torch.sqrt(y)
        y = torch.sqrt(y)

        return x/y.view(x.shape[0],1)


    """ Forward """
    def forward(self,feature,hidden):
        # Filtering
        _,(hf,_) = self.filteringLSTM(hidden)

        xfr = self.wfR(hf)      # [1,batch,a]*[a,b] = [1,batch,b]
        xfl = self.wfL(hidden)  # [1,batch,a]*[a,b] = [1,batch,b]
        
        xf = self.ReLu( xfl+xfr )   # [1,batch,b]*[b,L] = [1,batch,L]
        xf = xf.transpose(0,1)      # [1,batch,D] -> [batch,1,D] 

        # Pigeonholing
        _,(hp,_) = self.pigeonholingLSTM(hidden)

        xpr = self.wpR(hp)      # [1,batch,a]*[a,b] = [1,batch,b]
        xpl = self.wpL(hidden)  # [1,batch,a]*[a,b] = [1,batch,b]
        
        xp = self.ReLu( xpl+xpr )    # [1,batch,c]*[c,D] = [1,batch,D]
        xp = xp.squeeze(0)                      # [1,batch,D] -> [batch,D]    

        # Attention maps
        visual = self.  wf( feature )     # [batch,L,D]*[D,D] = [batch,L,D]
        visual = self.ReLu(  visual )

        featureFil = self.avgFiltering ( visual * xf ).squeeze(2)
        alpha = self.norm4( featureFil )    # [batch,L]
        
        featurePig = self.avgPigeonholing(feature.transpose(1,2)).squeeze(2)
        featurePig = self.Sigmoid( self.wp( featurePig ) )
        beta  = self.norm4( featurePig*xp )    # [batch,D]
        
        return alpha.unsqueeze(2), beta.unsqueeze(1)
        

class Atten13(nn.Module):
    """ Constructor """
    def __init__(self, cube_size, n_hidden):
        super(Atten13, self).__init__()
        # Parameters
        self.D = cube_size[2]               #  depth
        self.L = cube_size[0]*cube_size[1]  #  h x w
        self.R = self.L*self.D              #  L x D
        self.H = n_hidden                   #  hidden_size
        self.M = n_hidden                   #  hidden_size

        # Filtering 
        self.filteringLSTM = nn.LSTM( input_size = self.H, hidden_size = 512)
        self.wfL = nn.Linear(1024,self.D,bias=True )
        self.wfR = nn.Linear( 512,self.D,bias=False)

        self.avgFiltering = nn.AdaptiveAvgPool1d(1)
        
        # Pigeonholing 
        self.pigeonholingLSTM = nn.LSTM(input_size = self.H, hidden_size = 512)
        self.wpL = nn.Linear(1024,self.D,bias=True )
        self.wpR = nn.Linear( 512,self.D,bias=False)
        self.avgPigeonholing = nn.AdaptiveAvgPool1d(1)
        
        self.wp = nn.Linear(self.D,self.D,bias=True)

        # Initialization
        self.   filteringLSTM.reset_parameters()
        self.pigeonholingLSTM.reset_parameters()
        torch.nn.init.xavier_uniform_(self.wfL.weight)
        torch.nn.init.xavier_uniform_(self.wfR.weight)
        torch.nn.init.xavier_uniform_(self.wpL.weight)
        torch.nn.init.xavier_uniform_(self.wpR.weight)
        torch.nn.init.xavier_uniform_(self.wp .weight)

        self.ReLu    = nn.ReLU()
        self.ReLu6   = nn.ReLU6()
        self.Sigmoid = nn.Sigmoid()
        self.Tanh    = nn.Tanh()
        self.Softmax = nn.Softmax(1)
        self.BNormF  = nn.BatchNorm1d(self.L)
        self.BNormP  = nn.BatchNorm1d(self.D)


    def norm2(self,x):
        y = self.Tanh(x)**2
        y = y.mean(1) + 10**-6
        y = torch.sqrt(y)

        return x/y.view(x.shape[0],1)

    def norm4(self,x):
        x = self.Tanh(x)
        y = x**4
        y = y.sum(1) + 10**-12
        y = torch.sqrt(y)
        y = torch.sqrt(y)

        return x/y.view(x.shape[0],1)

    """ Forward """
    def forward(self,feature,hidden):
        # Filtering
        _,(hf,_) = self.filteringLSTM(hidden)

        xfr = self.wfR(hf)      # [1,batch,a]*[a,b] = [1,batch,b]
        xfl = self.wfL(hidden)  # [1,batch,a]*[a,b] = [1,batch,b]
        
        xf = self.ReLu( xfl+xfr )   # [1,batch,b]*[b,L] = [1,batch,L]
        xf = xf.transpose(0,1)      # [1,batch,D] -> [batch,1,D] 

        # Pigeonholing
        _,(hb,_) = self.pigeonholingLSTM(hidden)

        xpr = self.wpR(hb)      # [1,batch,h]*[h,D] = [1,batch,D]
        xpl = self.wpL(hidden)  # [1,batch,h]*[h,D] = [1,batch,D]
        
        xb = self.ReLu( xpl+xpr )   # [1,batch,D]
        xb = xb.squeeze(0)          # [1,batch,D] -> [batch,D,1] 

        # Attention maps
        featureFil = self.avgFiltering ( feature * xf ).squeeze(2)
        alpha = self.norm4( featureFil )    # [batch,L]

        _p   = self.Sigmoid( self.wp(feature) )         # [batch,L,D]*[D,D] = [batch,L,D]
        _p   = self.avgPigeonholing(_p.transpose(1,2) ) # [batch,L,D] -> [batch,L,1]
        _p   = _p.squeeze(2)                            # [batch,L,1] -> [batch,L]
        beta = self.norm4( _p*xb  )                     # [batch,D]
        
        return alpha.unsqueeze(2), beta.transpose(1,2) 
        

# ------------------------------------------------------------
class Atten14(nn.Module):
    """ Constructor """
    def __init__(self, cube_size, n_hidden):
        super(Atten14, self).__init__()
        # Parameters
        self.D = cube_size[2]               #  depth
        self.L = cube_size[0]*cube_size[1]  #  h x w
        self.R = self.L*self.D              #  L x D
        self.H = n_hidden                   #  hidden_size
        self.M = n_hidden                   #  hidden_size

        # Spatial 
        self.spatialLSTM = nn.LSTM( input_size = self.H, hidden_size = 512)
        self.convSC  = nn.Conv1d(1,1,kernel_size=11,bias=False,padding=5,stride=2)
        self.convSR  = nn.Conv1d(1,1,kernel_size= 5,bias=False,padding=2,stride=1)
        self.convS   = nn.Conv1d(1,1,kernel_size= 5,bias=False,padding=2,stride=2)
        self.maxpool = torch.nn.AdaptiveMaxPool1d(self.D)

        self.avgFiltering = nn.AdaptiveAvgPool1d(1)
        
        # Pigeonholing 
        self.pigeonholingLSTM = nn.LSTM(input_size = self.H, hidden_size = 512)
        self.wpL = nn.Linear(1024,self.D,bias=True )
        self.wpR = nn.Linear( 512,self.D,bias=False)
        self.avgPigeonholing = nn.AdaptiveAvgPool1d(1)
        
        self.wp = nn.Linear(self.D,self.D,bias=True)

        # Initialization
        self.     spatialLSTM.reset_parameters()
        self.pigeonholingLSTM.reset_parameters()
        torch.nn.init.xavier_uniform_(self.convSC.weight)
        torch.nn.init.xavier_uniform_(self.convSR.weight)
        torch.nn.init.xavier_uniform_(self.convS.weight)
        torch.nn.init.xavier_uniform_(self.wpL.weight)
        torch.nn.init.xavier_uniform_(self.wpR.weight)
        torch.nn.init.xavier_uniform_(self.wp .weight)

        self.ReLu    = nn.ReLU()
        self.ReLu6   = nn.ReLU6()
        self.Sigmoid = nn.Sigmoid()
        self.Tanh    = nn.Tanh()
        self.Softmax = nn.Softmax(1)
        self.BNormF  = nn.BatchNorm1d(self.L)
        self.BNormP  = nn.BatchNorm1d(self.D)


    def norm2(self,x):
        y = self.Tanh(x)**2
        y = y.mean(1) + 10**-6
        y = torch.sqrt(y)

        return x/y.view(x.shape[0],1)

    def norm4(self,x):
        x = self.Tanh(x)
        y = x**4
        y = y.sum(1) + 10**-12
        y = torch.sqrt(y)
        y = torch.sqrt(y)

        return x/y.view(x.shape[0],1)


    """ Forward """
    def forward(self,feature,hidden):
        # Spatial Attention
        _,(hs,_) = self.spatialLSTM(hidden)
        hc = hidden.transpose(0,1)
        hs =   hs  .transpose(0,1)

        xsc = self.convSC( hc)     # [batch,1,a] -> [batch,1,a]
        xsc = self.ReLu(xsc)

        xsr = self.convSR(hs)       # [batch,1,b] -> [batch,1,b]
        xsr = self.ReLu(xsr)

        xs = self.convS(xsc + xsr)  # [batch,1,b] -> [batch,1,b]
        xs = self.ReLu(xs)
        xs = self.maxpool(xs)       # [batch,1,b] -> [batch,1,D]
         
        featureFil = self.avgFiltering ( feature * xs ).squeeze(2)
        alpha = self.Softmax( featureFil )    # [batch,L]

        # Pigeonholing
        _,(hp,_) = self.pigeonholingLSTM(hidden)

        xpr = self.wpR(hp)      # [1,batch,a]*[a,b] = [1,batch,b]
        xpl = self.wpL(hidden)  # [1,batch,a]*[a,b] = [1,batch,b]
        
        xp = self.ReLu( xpl+xpr )    # [1,batch,c]*[c,D] = [1,batch,D]
        xp = xp.squeeze(0)                      # [1,batch,D] -> [batch,D]    

        # Attention maps
        featurePig = self.avgPigeonholing(feature.transpose(1,2)).squeeze(2)
        featurePig = self.Sigmoid( self.wp( featurePig ) )
        beta  = self.norm4( featurePig*xp )    # [batch,D]
        
        return alpha.unsqueeze(2), beta.unsqueeze(1)






# ------------------------------------------------------------
class SpatialAttnNet(nn.Module):
    """ Constructor """
    def __init__(self, cube_size):
        super(SpatialAttnNet, self).__init__()
        # Parameters 
        self.high  = cube_size[0]
        self.width = cube_size[1]
        self.D = cube_size[2]           #  depth
        self.L = self.high*self.width   #  h x w
        self.R = self.L*self.D          #  L x D
        
        # Deberian ser entrada
        self.batch = 120
        self.M = 64

        self.h =  2
        self.d = int(self.D/self.h)
        self.hd = self.d*self.h
        self.sqrtd = self.d ** .5

        # Spatial 
        self.to_q = nn.Conv2d(self.D, self.hd, 1, bias = False)
        self.to_v = nn.Conv2d(self.D, self.hd, 1, bias = False)
        # self.to_q = nn.Linear(self.D, self.hd, bias = False)
        # self.to_v = nn.Linear(self.D, self.hd, bias = False)
        self.to_k = nn.Linear(self.M, self.hd, bias = False)

        self.fc = nn.Linear(self.hd, self.D)

        self.norm_q = nn.BatchNorm2d(self.d*self.h)
        self.norm_v = nn.BatchNorm2d(self.d*self.h)

        # Initialization
        torch.nn.init.xavier_uniform_(self.to_q.weight)
        torch.nn.init.xavier_uniform_(self.to_k.weight)
        torch.nn.init.xavier_uniform_(self.to_v.weight)
        torch.nn.init.xavier_uniform_(self. fc .weight)

        self.ReLu    = nn.ReLU()
        self.Softmax = nn.Softmax(3)


    def norm4(self,x):
        y = self.Tanh(x)**4
        y = y.mean(1) + 10**-12
        y = torch.sqrt(y)
        y = torch.sqrt(y)

        return x/y.view(x.shape[0],1)

    """ Forward 
          - eta [batch,channel,high,width]
          -  F  [batch,1,M]
    """
    def forward(self,ηt,Ft0):
        # Visual feature
        # ηt = ηt.view(-1, self.D, self.L)
        # ηt = ηt.transpose(1,2)    # [batch,L,D]
        
        Q = self.to_q(ηt )     # [batch,hd,L]
        K = self.to_k(Ft0)     # [batch,L,hd]
        V = self.to_v(ηt )     # [batch,hd,L]

        Q = self.norm_q(Q)
        V = self.norm_v(V)

        Q = Q.view(-1, self.hd, self.L)#.transpose(1,2)  # [batch,L,hd]
        V = V.view(-1, self.hd, self.L)#.transpose(1,2)  # [batch,L,hd]

        Q, K, V = map(lambda x: x.reshape(x.shape[0], self.h, self.d, -1), [Q,K,V])  # [batch,h,d,L]
        # Q, K, V = map(lambda x: x.reshape(x.shape[0], -1, self.h, self.d), [Q,K,V])  # [batch,L,h,d]
        # QK = torch.einsum('bnhd,bmhd->bhnm', (Q,K))
        QK = torch.einsum('bhdn,bhdm->bhnm', (Q,K))

        # Attention map
        A  = self.Softmax(QK/self.sqrtd)                # [batch, h, L, 1]
        # Z  = torch.einsum('bhnk,bnhd->bnhd', (A,V))     # [batch, L, h, d]
        Z  = torch.einsum('bhnk,bhdn->bnhd', (A,V))     # [batch, L, h, d]
        
        # Output
        Z = Z.view(-1,self.L,self.hd)
        Z = self.fc(Z)          # [batch, L, D]
        Z = Z.transpose(1,2)    # [batch, D, L]

        return Z.reshape(-1,self.D,self.high,self.width)
        

class FeatureAttnNet(nn.Module):
    """ Constructor """
    def __init__(self, n_encode, n_hidden, n_command):
        super(FeatureAttnNet, self).__init__()
        self.n_features = 32
        self.n_depth    = 64
        self.sqrtDepth  = math.sqrt(self.n_depth)

        self.M = self.n_depth*int(self.n_features/2)

        # Feature 
        self.to_q  = nn.Linear(n_command, self.n_depth, bias = False)
        self.to_kz = nn.Linear( n_encode, self.   M   , bias = False)
        self.to_kh = nn.Linear( n_hidden, self.   M   , bias = False)
        self.to_vz = nn.Linear( n_encode, self.   M   , bias = False)
        self.to_vh = nn.Linear( n_hidden, self.   M   , bias = False)

        self.Softmax = nn.Softmax(2)

        # Batch normalization
        self.normQ = nn.BatchNorm1d(64)
        self.normK = nn.BatchNorm1d(64)
        self.normV = nn.BatchNorm1d(64)

        # Initialization
        torch.nn.init.xavier_uniform_(self.to_q .weight)
        torch.nn.init.xavier_uniform_(self.to_kz.weight)
        torch.nn.init.xavier_uniform_(self.to_kh.weight)
        torch.nn.init.xavier_uniform_(self.to_vz.weight)
        torch.nn.init.xavier_uniform_(self.to_vh.weight)

    def forward(self,feature,hidden,command):
        # Query
        Q = self.to_q(command)
        Q = Q.unsqueeze(1)
        Q = self.normQ(Q.transpose(1,2))   # [120,1,64]

        # Key
        Kz = self.to_kz(feature) 
        Kh = self.to_kh( hidden)
        K = torch.cat([Kz,Kh],dim=1)
        K = K.view(-1,self.n_features,self.n_depth)
        K = self.normK(K.transpose(1,2))   # [120,32,64]

        # Value
        Vz = self.to_vz(feature)
        Vh = self.to_vh(hidden)
        V = torch.cat([Vz,Vh],dim=1)
        V = V.view(-1,self.n_features,self.n_depth)

        # Attention 
        A = torch.matmul(Q.transpose(1,2),K)
        A = self.Softmax(A/self.sqrtDepth)   # [120,1,16]

        s = torch.matmul(A,V)   # [120,1,64]
        return s.view(-1,64)#.squeeze()
        

class CommandNet(nn.Module):
    """ Constructor """
    def __init__(self,n_encode=16):
        super(CommandNet, self).__init__()
        self.Wc = nn.Linear( 4, n_encode, bias= True)
        self.ReLU    = nn.ReLU()

    def forward(self,control):
        c = control*2-1
        c = self.Wc(c)
        return self.ReLU(c)
        
