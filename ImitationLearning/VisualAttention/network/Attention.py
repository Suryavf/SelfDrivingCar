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
        y = self.Tanh(x)**4
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
        y = self.Tanh(x)**4
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
        y = self.Tanh(x)**4
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
        y = self.Tanh(x)**4
        y = y.mean(1) + 10**-12
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
class SpatialAttention(nn.Module):
    """ Constructor """
    def __init__(self, cube_size, n_hidden):
        super(SpatialAttention, self).__init__()
        # Parameters
        D = cube_size[2]                #  depth
        L = cube_size[0]*cube_size[1]   #  h x w
        R = self.L*self.D               #  L x D
        hc =     n_hidden               #  hidden_size
        hs = int(n_hidden/2)            #  hidden_size

        # Spatial 
        self.attnLSTM = nn.LSTM( input_size=hc, hidden_size=hs )
        self.convSC   = nn.Conv1d(1,1,kernel_size=11,bias=False,padding=5,stride=2)
        self.convSR   = nn.Conv1d(1,1,kernel_size= 5,bias=False,padding=2,stride=1)
        self.convS    = nn.Conv1d(1,1,kernel_size= 5,bias=False,padding=2,stride=2)
        self.maxpool  = torch.nn.AdaptiveMaxPool1d(D)

        self.average = nn.AdaptiveAvgPool1d(1)
        
        # Initialization
        self.attnLSTM.reset_parameters()
        torch.nn.init.xavier_uniform_(self.convSC.weight)
        torch.nn.init.xavier_uniform_(self.convSR.weight)
        torch.nn.init.xavier_uniform_(self.convS.weight)

        self.ReLu    = nn.ReLU()
        self.Softmax = nn.Softmax(1)


    def norm4(self,x):
        y = self.Tanh(x)**4
        y = y.mean(1) + 10**-12
        y = torch.sqrt(y)
        y = torch.sqrt(y)

        return x/y.view(x.shape[0],1)

    """ Forward """
    def forward(self,feature,hidden):
        # Spatial Attention
        _,(hs,_) = self.spatialLSTM(hidden)
        hc = hidden.transpose(0,1)
        hs =   hs  .transpose(0,1)

        xsc = self.ReLu(self.convSC(hc))   # [batch,1,a] -> [batch,1,a]
        xsr = self.ReLu(self.convSR(hs))   # [batch,1,b] -> [batch,1,b]

        xs = self.ReLu(self.convS(xsc + xsr))   # [batch,1,b] -> [batch,1,b]
        xs = self.maxpool(xs)                   # [batch,1,b] -> [batch,1,D]
         
        alpha = self.average( feature * xs ).squeeze(2)
        alpha = self.Softmax( alpha )       # [batch,L]
        
        return alpha.unsqueeze(2)
        
