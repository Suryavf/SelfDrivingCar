import torch
import torch.nn as nn
from IPython.core.debugger import set_trace

""" Basic Decoder Module
    --------------------
    Ref: Kim, J., & Canny, J. (2017). "Interpretable learning for self-driving 
         cars by visualizing causal attention". In Proceedings of the IEEE 
         international conference on computer vision (pp. 2942-2950).
    
        * Input: feature [batch,L,D]
                 hidden  [1,batch,H]
        * Output: alpha  [batch,L,1]
"""
class BasicDecoder(nn.Module):
    """ Constructor """
    def __init__(self,AttentionNet,ControlNet,cube_size,n_hidden):
        super(BasicDecoder, self).__init__()
        
        # Parameters
        self.D = cube_size[2]               #   64 
        self.L = cube_size[0]*cube_size[1]  #   90 
        self.R = self.L*self.D              # 5760 
        self.H = n_hidden                   #  512 
        self.M = n_hidden                   #  512 
        self.sequence_len =  20
        self.batch_size   = 120
        self.n_out        =   3
        
        # Output container
        self.pred = torch.zeros([self.batch_size,self.n_out])
        self. map = torch.zeros([self.batch_size,self.  L  ])

        # Declare layers
        self.attn = AttentionNet
        self.ctrl =   ControlNet
        self.lstm = nn.LSTM(  input_size = self.R,
                             hidden_size = self.H,
                              num_layers =      1)
        
        self.init_Wh   = nn.Linear(self.D,self.H,bias=True )
        self.init_Wc   = nn.Linear(self.D,self.H,bias=True )
        self.init_tanh = nn.Tanh()

        # Initialization
        torch.nn.init.xavier_uniform_(self.init_Wh.weight)
        torch.nn.init.xavier_uniform_(self.init_Wc.weight)
        self.lstm.reset_parameters()
        
        
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
        # Parameters
        sequence_len = self.sequence_len
        n_out        = self.n_out
        if self.training: batch_size = int(feature.shape[0]/sequence_len)
        else            : batch_size =     feature.shape[0]
        
        # Data
        if self.training: sequence = feature.view(batch_size,sequence_len,self.L,self.D).transpose_(0,1) # [sequence,batch,L,D]
        else            : sequence = feature  # [batch,L,D]
        
        # Input to inicialize LSTM 
        if self.training: xt = sequence[0]
        else            : xt = sequence[0].unsqueeze(0)

        # Inicialize hidden state and cell state
        #   * hidden ~ [1,batch,H]
        #   * cell   ~ [1,batch,H]
        hidden,cell = self.initializeLSTM(xt)

        # Prediction container
        if self.training:
            self.pred = self.pred.view(sequence_len,batch_size, n_out)
            self. map = self. map.view(sequence_len,batch_size,self.L)

        # Sequence loop
        n_range  = self.sequence_len if self.training else batch_size
        n_visual =        batch_size if self.training else     1
        for k in range(n_range):
            # One time
            if self.training: xt = sequence[k]              # [batch,L,D]
            else            : xt = sequence[k].unsqueeze(0) # [  1  ,L,D]
            
            # Visual Attention
            alpha = self.attn(xt,hidden)# [batch,L,1]
            visual = xt * alpha                         # [batch,L,D]x[batch,L,1] = [batch,L,D]
            
            visual = visual.reshape(n_visual,self.R)    # [batch,R]
            visual = visual.unsqueeze(0)                # [1,batch,R]

            # LSTM
            #  * yt     ~ [sequence,batch,H]
            #  * hidden ~ [ layers ,batch,H]
            #  * cell   ~ [ layers ,batch,H]
            _,(hidden,cell) = self.lstm(visual,(hidden,cell))
            
            # Control
            self.pred[k] = self.ctrl(visual, hidden, training=self.training)
            self. map[k] = alpha.squeeze()

        if self.training: 
            self.pred = self.pred.transpose(0,1).contiguous().view(batch_size*sequence_len, n_out)
            self. map = self. map.transpose(0,1).contiguous().view(batch_size*sequence_len,self.L)
        
        return self.pred, self.map


""" Dual Decoder Module
    -------------------
        * Input: feature [batch,L,D]
                 hidden  [1,batch,H]
        * Output: alpha  [batch,L,1]
"""
class DualDecoder(nn.Module):
    """ Constructor """
    def __init__(self,AttentionNet,ControlNet,cube_size,n_hidden):
        super(DualDecoder, self).__init__()
        
        # Parameters
        self.D = cube_size[2]               #   64 # cube_size[0]
        self.L = cube_size[0]*cube_size[1]  #   90 # cube_size[1]*cube_size[2]
        self.R = self.L*self.D              # 5760 # self.L*self.D
        self.H = n_hidden                   # 1024 # hidden_size
        self.M = n_hidden                   # 1024 # hidden_size
        self.sequence_len =  20
        self.batch_size   = 120
        self.n_out        =   3
        self.study = False

        # Declare layers
        self.attn = AttentionNet
        self.ctrl =   ControlNet
        self.lstm = nn.LSTM( input_size=self.R, hidden_size=self.H, num_layers=2 )
        
        self.init_Wh1  = nn.Linear(self.D,self.H,bias=True )
        self.init_Wh2  = nn.Linear(self.H,self.H,bias=True )
        self.init_Wc1  = nn.Linear(self.D,self.H,bias=True )
        self.init_Wc2  = nn.Linear(self.H,self.H,bias=True )
        self.init_tanh = nn.Tanh()

        # Initialization
        torch.nn.init.xavier_uniform_(self.init_Wh1.weight)
        torch.nn.init.xavier_uniform_(self.init_Wh2.weight)
        torch.nn.init.xavier_uniform_(self.init_Wc1.weight)
        torch.nn.init.xavier_uniform_(self.init_Wc2.weight)
        self.lstm.reset_parameters()
        

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
            
            hidden1 = self.init_Wh1 (feature) # [batch,D]*[D,H] -> [batch,H]
            hidden1 = self.init_tanh(hidden1) # [batch,H]
            hidden2 = self.init_Wh2 (hidden1) # [batch,H]*[H,H] -> [batch,H]
            hidden2 = self.init_tanh(hidden2) # [batch,H]

            cell1 = self.init_Wc1 (feature)   # [batch,D]*[D,H] -> [batch,H]
            cell1 = self.init_tanh( cell1 )   # [batch,H]
            cell2 = self.init_Wc2 ( cell1 )   # [batch,H]*[H,H] -> [batch,H]
            cell2 = self.init_tanh( cell2 )   # [batch,H]

            hidden = torch.stack([hidden1,hidden2], dim=0).contiguous() # [2,batch,H]
            cell   = torch.stack([  cell1,  cell2], dim=0).contiguous() # [2,batch,H]

            # (h,c) ~ [num_layers, batch, hidden_size]
            return hidden,cell
        

    """ Forward """
    def forward(self,feature):
        # Parameters
        sequence_len = self.sequence_len
        n_out        = self.n_out
        if self.training: batch_size = int(feature.shape[0]/sequence_len)
        else            : batch_size =     feature.shape[0]
        
        # Data
        if self.training: sequence = feature.view(batch_size,sequence_len,self.L,self.D).transpose_(0,1) # [sequence,batch,L,D]
        else            : sequence = feature  # [batch,L,D]
        
        # Input to inicialize LSTM 
        if self.training: xt = sequence[0]
        else            : xt = sequence[0].unsqueeze(0)

        # Inicialize hidden state and cell state
        #   * hidden ~ [1,batch,H]
        #   * cell   ~ [1,batch,H]
        hidden,cell = self.initializeLSTM(xt)

        # Prediction container
        if self.training:
            pred = torch.zeros([sequence_len,batch_size, n_out]).to( torch.device('cuda:0') )
            map_ = torch.zeros([sequence_len,batch_size,self.L]).to( torch.device('cuda:0') )
        else:
            pred = torch.zeros([batch_size, n_out]).to( torch.device('cuda:0') )
            map_ = torch.zeros([batch_size,self.L]).to( torch.device('cuda:0') )

        # Study rountime
        if self.study:
            action = torch.zeros([self.batch_size,self.H]).to( torch.device('cuda:0') )
            atten  = torch.zeros([self.batch_size,self.H]).to( torch.device('cuda:0') )
        else:
            action,atten = (None,None)

        # Sequence loop
        n_range  = self.sequence_len if self.training else batch_size
        n_visual =        batch_size if self.training else     1
        for k in range(n_range):
            # One time
            if self.training: xt = sequence[k]              # [batch,L,D]
            else            : xt = sequence[k].unsqueeze(0) # [  1  ,L,D]
            
            # Visual Attention
            alpha = self.attn(xt,hidden[1].unsqueeze(0))# [batch,L,1]
            visual = xt * alpha                         # [batch,L,D]x[batch,L,1] = [batch,L,D]
            
            visual = visual.reshape(n_visual,self.R)    # [batch,R]
            visual = visual.unsqueeze(0)                # [1,batch,R]
            
            # LSTM
            #  * yt     ~ [sequence,batch,H]
            #  * hidden ~ [ layers ,batch,H]
            #  * cell   ~ [ layers ,batch,H]
            _,(hidden,cell) = self.lstm(visual,(hidden,cell))
            
            # Control
            pred[k] = self.ctrl(visual, hidden[0].unsqueeze(0))
            map_[k] = alpha.squeeze()
            if self.study:
                action[k] = hidden[0].squeeze()
                atten [k] = hidden[1].squeeze()

        if self.training: 
            pred = pred.transpose(0,1).contiguous().view(batch_size*sequence_len, n_out)
            map_ = map_.transpose(0,1).contiguous().view(batch_size*sequence_len,self.L)
        
        # Return
        return pred, map_, {'action': action, 'attention': atten}
        

""" TVA Decoder Module
    ------------------
    Ref: Kim, J., & Canny, J. (2017). "Interpretable learning for self-driving 
         cars by visualizing causal attention". In Proceedings of the IEEE 
         international conference on computer vision (pp. 2942-2950).
    
        * Input: feature [batch,L,D]
                 hidden  [1,batch,H]
        * Output: alpha  [batch,L,1]
"""
class TVADecoder(nn.Module):
    """ Constructor """
    def __init__(self,AttentionNet,cube_size,n_hidden):
        super(TVADecoder, self).__init__()
        
        # Parameters
        self.D = cube_size[2]               #   64 
        self.L = cube_size[0]*cube_size[1]  #   90 
        self.R = self.L*self.D              # 5760 
        self.H = n_hidden                   #  512 
        self.M = n_hidden                   #  512 
        self.sequence_len =  20
        self.batch_size   = 120
        self.n_out        =   3
        self.study = False
        
        # Declare layers
        self.attn = AttentionNet
        self.lstm = nn.LSTM(  input_size = self.R,
                             hidden_size = self.H,
                              num_layers =      1)
        
        self.init_Wh   = nn.Linear(self.D,self.H,bias=True )
        self.init_Wc   = nn.Linear(self.D,self.H,bias=True )
        self.init_tanh = nn.Tanh()

        # Initialization
        torch.nn.init.xavier_uniform_(self.init_Wh.weight)
        torch.nn.init.xavier_uniform_(self.init_Wc.weight)
        self.lstm.reset_parameters()
        
        
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
        # Parameters
        sequence_len = self.sequence_len
        if self.training: batch_size = int(feature.shape[0]/sequence_len)
        else            : batch_size =     feature.shape[0]
        
        # Data
        if self.training: sequence = feature.view(batch_size,sequence_len,self.L,self.D).transpose_(0,1) # [sequence,batch,L,D]
        else            : sequence = feature  # [batch,L,D]
        
        # Input to inicialize LSTM 
        if self.training: xt = sequence[0]
        else            : xt = sequence[0].unsqueeze(0)

        # Inicialize hidden state and cell state
        #   * hidden ~ [1,batch,H]
        #   * cell   ~ [1,batch,H]
        hidden,cell = self.initializeLSTM(xt)

        # Prediction container
        if self.training:
            vis_ = torch.zeros([sequence_len,batch_size,self.R]).to( torch.device('cuda:0') )
            alp_ = torch.zeros([sequence_len,batch_size,self.L]).to( torch.device('cuda:0') )
            bet_ = torch.zeros([sequence_len,batch_size,self.D]).to( torch.device('cuda:0') )
            hdd_ = torch.zeros([sequence_len,batch_size,self.H]).to( torch.device('cuda:0') )
        else:
            vis_ = torch.zeros([batch_size,self.R]).to( torch.device('cuda:0') )
            alp_ = torch.zeros([batch_size,self.L]).to( torch.device('cuda:0') )
            bet_ = torch.zeros([batch_size,self.D]).to( torch.device('cuda:0') )
            hdd_ = torch.zeros([batch_size,self.H]).to( torch.device('cuda:0') )

        # Study rountime
        if self.study:
            hc = torch.zeros([self.batch_size,self.H]).to( torch.device('cuda:0') )
        else:
            hc = None # hc,ha,hb = (None,None,None)

        # Sequence loop
        n_range  = self.sequence_len if self.training else batch_size
        n_visual =        batch_size if self.training else     1
        for k in range(n_range):
            # One time
            if self.training: xt = sequence[k]              # [batch,L,D]
            else            : xt = sequence[k].unsqueeze(0) # [  1  ,L,D]
            
            # Visual Attention
            alpha,beta = self.attn(xt,hidden)  # [batch,L,1]

            # Spatial
            spatial =  xt * alpha      # [batch,L,D]x[batch,L,1] = [batch,L,D]
            visual  = spatial + xt

            # Categorical
            visual = visual * beta      # [batch,L,D]x[batch,1,D] = [batch,L,D]

            visual = visual.reshape(n_visual,self.R)    # [batch,R]
            visual = visual.unsqueeze(0)                # [1,batch,R]

            # LSTM
            #  * yt     ~ [sequence,batch,H]
            #  * hidden ~ [ layers ,batch,H]
            #  * cell   ~ [ layers ,batch,H]
            _,(hidden,cell) = self.lstm(visual,(hidden,cell))
            
            # Output
            vis_[k] = visual                    # [1,batch,R]
            alp_[k] =  alpha.squeeze()          # [1,batch,L]
            bet_[k] =   beta.squeeze()          # [1,batch,D]
            hdd_[k] = hidden[0].unsqueeze(0)    # [1,batch,H]

            if self.study:
                hc[k] = hidden[0].squeeze()

        if self.training: 
            vis_ = vis_.transpose(0,1).contiguous().view(batch_size*sequence_len,self.R)
            alp_ = alp_.transpose(0,1).contiguous().view(batch_size*sequence_len,self.L)
            bet_ = bet_.transpose(0,1).contiguous().view(batch_size*sequence_len,self.D)
            hdd_ = hdd_.transpose(0,1).contiguous().view(batch_size*sequence_len,self.H)

        return vis_, hdd_, {'alpha': alp_, 'beta': bet_}, {'control': hc}
        

# ------------------------------------------------------------------------------------------------
#
#
# ------------------------------------------------------------------------------------------------
class CatDecoder(nn.Module):
    """ Constructor """
    def __init__(self,  HighEncoderNet, SpatialNet, FeatureNet, CommandNet, 
                        LowLevelDim=128, HighLevelDim=512, 
                        n_hidden=1024, n_state=64,n_task=3):
        super(CatDecoder, self).__init__()
        self.study = False

        # Parameters
        self.H =     n_hidden       # output LSTM   1024   2048
        self.R = int(n_hidden/4)    #  input LSTM    256    512
        self.S =     n_state
        self.n_task       =  n_task
        self.sequence_len =      20
        
        # Attention
        self.HighEncoder = HighEncoderNet
        self.SpatialAttn =     SpatialNet
        self.FeatureAttn =     FeatureNet
        self. CmdDecoder =     CommandNet

        # Output
        self.dimReduction = nn.Conv2d(HighLevelDim,self.R, kernel_size=1, bias=False)
        self.lstm = nn.LSTM(  input_size = self.R,
                             hidden_size = self.H,
                              num_layers =      1)
        self.init_Wh   = nn.Linear(LowLevelDim,self.H,bias=True )
        self.init_Wc   = nn.Linear(LowLevelDim,self.H,bias=True )
        self.init_tanh = nn.Tanh()

        self.avgpool1 = nn.AdaptiveAvgPool2d((1, 1))
        self.avgpool2 = nn.AdaptiveAvgPool2d((1, 1))
        self.normSpa  = nn.BatchNorm2d(128)
        self.ReLU     = nn.ReLU()  

        # Initialization
        torch.nn.init.xavier_uniform_(self.dimReduction.weight)
        torch.nn.init.xavier_uniform_(self.     init_Wh.weight)
        torch.nn.init.xavier_uniform_(self.     init_Wc.weight)
        self.lstm.reset_parameters()
        
        
    """ Initialize LSTM: hidden state and cell state
        Ref: Xu, Kelvin, et al. "Show, attend and tell: Neural 
            image caption generation with visual attention." 
            International conference on machine learning. 2015.

            * Input:  feature [batch,D,h,w]
            * Output: hidden  [1,batch,H]
                      cell    [1,batch,H]
    """
    def initializeLSTM(self,feature):
        with torch.no_grad():
            # Mean features
            feature = torch.mean(feature,(2,3)) # [batch,D,h,w] -> [batch,D]
            
            hidden = self.init_Wh(feature)  # [batch,D]*[D,H] -> [batch,H]
            hidden = self.init_tanh(hidden) # [batch,H]
            hidden = hidden.unsqueeze(0)    # [1,batch,H]

            cell = self.init_Wc(feature)    # [batch,D]*[D,H] -> [batch,H]
            cell = self.init_tanh(cell)     # [batch,H]
            cell = cell.unsqueeze(0)        # [1,batch,H]

            # (h,c) ~ [num_layers, batch, hidden_size]
            return hidden.contiguous(),cell.contiguous()
        
    """ Forward 
          - eta [batch,channel,high,width]
    """
    def forward(self,feature,command):
        # Parameters
        sequence_len = self.sequence_len
        if self.training: batch_size = int(feature.shape[0]/sequence_len)
        else            : batch_size =     feature.shape[0]
        _,C,H,W = feature.shape # Batch of Tensor Images is a tensor of (B, C, H, W) shape
        
        # Data
        if self.training: sequence = feature.view(batch_size,sequence_len,C,H,W).transpose(0,1) # [sequence,batch, ...]
        else            : sequence = feature  # [batch, ...]
        
        # Inicialize hidden state and cell state
        #   * hidden ~ [1,batch,H]
        #   * cell   ~ [1,batch,H]
        if self.training: xt = sequence[0]  # Input to inicialize LSTM 
        else            : xt = sequence[0].unsqueeze(0)
        ht,ct = self.initializeLSTM(xt)

        # Command decoder
        cmd = self.CmdDecoder(command)
        if self.training: cmd = cmd.view(batch_size,sequence_len,-1).transpose(0,1) # [sequence,batch,4]

        # Prediction container
        if self.training:
            st_ = torch.zeros([sequence_len,batch_size,self.n_task,self.S]).to( torch.device('cuda:0') )
            ht_ = torch.zeros([sequence_len,batch_size,            self.H]).to( torch.device('cuda:0') )
        else:
            st_ = torch.zeros([batch_size,self.n_task,self.S]).to( torch.device('cuda:0') )
            ht_ = torch.zeros([batch_size,            self.H]).to( torch.device('cuda:0') )

        # State initialization
        if self.training: st = torch.rand([batch_size,self.n_task,self.S]).to( torch.device('cuda:0') )
        else            : st = torch.rand([         1,self.n_task,self.S]).to( torch.device('cuda:0') )
        
        # Study
        if self.study:
            α,β,F = list(),list(),list()
        else:
            α,β,F = None,None,None

        # Sequence loop
        n_range  = self.sequence_len if self.training else batch_size
        for k in range(n_range):
            # One time
            if self.training: ηt = sequence[k]              # [batch,L,D]
            else            : ηt = sequence[k].unsqueeze(0) # [  1  ,L,D]
            if self.training: cm = cmd[k]                   # [batch, 4 ]
            else            : cm = cmd[k].unsqueeze(0)      # [  1  , 4 ]

            # Spatial Attention
            xt, αt = self.SpatialAttn(ηt,st)
            xt = self.normSpa(xt)
            xt = self.ReLU(ηt + xt)
            
            # High-level encoder
            zt = self.HighEncoder(xt)

            # Feature-based attention
            # s[t] = f(z[t],h[t-1])
            _zt = self.avgpool1( zt)
            _zt = torch.flatten(_zt, 1)
            st, ft, βt = self.FeatureAttn(_zt,ht[0],cm) # [batch,S]
            
            # Dimension reduction to LSTM
            rt = self.dimReduction(zt)
            rt = self.    avgpool2(rt)
            rt = torch.flatten(rt , 1)
            rt = rt.unsqueeze(0)
            
            # LSTM
            #  * yt     ~ [sequence,batch,H]
            #  * hidden ~ [ layers ,batch,H]
            #  * cell   ~ [ layers ,batch,H]
            _,(ht,ct)= self.lstm(rt,(ht,ct))
            
            # Output
            st_[k] = st.unsqueeze(0)    # [1,batch,S]
            ht_[k] = ht#.unsqueeze(0)   # [1,batch,H]

            # Study
            if self.study:
                α.append(αt)
                β.append(βt)
                F.append(ft)

        if self.training: 
            st_ = st_.transpose(0,1).reshape(batch_size*sequence_len,self.n_task,self.S).contiguous()
            ht_ = ht_.transpose(0,1).reshape(batch_size*sequence_len,self.n_task,self.S).contiguous()
        
        # Compile study
        if self.study:
            α = torch.cat(α, dim=0)
            β = torch.cat(β, dim=0)
            F = torch.cat(F, dim=0)
        
        return st_, ht_, {'alpha': α, 'beta': β}, {'features': F}
        
