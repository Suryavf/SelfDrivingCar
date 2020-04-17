import math 
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ref https://github.com/epfml/attention-cnn/blob/master/models/bert.py
def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

# Transformer
# http://jalammar.github.io/illustrated-transformer/
# https://arxiv.org/pdf/1706.03762.pdf
# https://towardsdatascience.com/how-to-code-the-transformer-in-pytorch-24db27c8f9ec
# https://ufal.mff.cuni.cz/pbml/110/art-popel-bojar.pdf

""" Self-Attention
    --------------
        * Attention function:
            A = Softmax [ QK'/sqrt_dk ] V
    Ref: Vaswani A., et al. (2017). "Attention is all you need". In Advances 
         in neural information processing systems (pp. 5998-6008).
"""
class SelfAttention(nn.Module):
    """ Constructor """
    def __init__(self, n_features,n_heads):
        super(SelfAttention, self).__init__()
        # Parameters
        self.dmodel  = n_features
        self.n_heads = n_heads
        self.dv = self.dmodel//n_heads
        self.dk = self.dv
        self.sqrt_dk = math.sqrt(self.dk)

        # Declare layers
        self.wq = nn.Linear(self.dmodel,self.dk,bias=True)
        self.wk = nn.Linear(self.dmodel,self.dk,bias=True)
        self.wv = nn.Linear(self.dmodel,self.dv,bias=True)

        # Initialization
        torch.nn.init.xavier_uniform_(self.wq.weight)
        torch.nn.init.xavier_uniform_(self.wk.weight)
        torch.nn.init.xavier_uniform_(self.wv.weight)
    
    """ Attention function """
    def attnFunc(self,Q,K,V):
        A = torch.matmul(Q,K.transpose(-2,-1))
        A = A/self.sqrt_dk
        A = nn.functional.softmax(A,dim=-1)
        A = torch.matmul(A,V)
        return A

    """ Forward """
    def forward(self,feature):
        Q = self.wq(feature)
        K = self.wk(feature)
        V = self.wv(feature)

        return self.attnFunc(Q,K,V)


""" Relative Positional Self-Attention
    ----------------------------------
        * Attention function:
            A = Softmax [ 1/sqrt_dk (QK'+ Sh + Sw) ] V
            
    Ref: Cheng-Zhi Anna Huang et al. (2018). "Music transformer: Generating 
         music with long-term structure". arXiv preprint arXiv:1809.04281.
         Bello, I., Zoph, B., Vaswani, A., Shlens, J., & Le, Q. V. (2019). 
         "Attention augmented convolutional networks". In Proceedings of the 
         IEEE International Conference on Computer Vision (pp. 3286-3295).
"""
class RelativeAttention(nn.Module):
    """ Constructor """
    def __init__(self, n_features,n_heads):
        super(RelativeAttention, self).__init__()
        # Parameters
        self.dmodel  = n_features
        self.n_heads = n_heads
        self.dv = self.dmodel//n_heads
        self.dk = self.dv
        self.sqrt_dk = math.sqrt(self.dk)

        # Declare layers
        self.wq = nn.Linear(self.dmodel,self.dk,bias=True)
        self.wk = nn.Linear(self.dmodel,self.dk,bias=True)
        self.wv = nn.Linear(self.dmodel,self.dv,bias=True)

        # Initialization
        torch.nn.init.xavier_uniform_(self.wq.weight)
        torch.nn.init.xavier_uniform_(self.wk.weight)
        torch.nn.init.xavier_uniform_(self.wv.weight)

    """ Attention function """
    def attnFunc(self,Q,K,Sh,Sw,V):
        A = torch.matmul(Q,K.transpose(-2,-1))
        A = A + Sh + Sw
        A = A/self.sqrt_dk
        A = nn.functional.softmax(A,dim=-1)
        A = torch.matmul(A,V)
        return A
    
    """ Forward """
    def forward(self,feature):
        pass


"""
    Multi-Head Attention layer
"""
class MultiHeadAttention(nn.Module):
    """ Constructor """
    def __init__(self, n_features, n_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.dmodel  = n_features
        self.n_heads = n_heads
        self.dv      = self.dmodel/n_heads
        self.dk      = self.dv
        self.dropout = dropout

        # Attention modules
        self.attns = [SelfAttention(n_features,n_heads) for _ in range(n_heads)]
        self.wo = nn.Linear(self.n_heads*self.dv,self.dmodel,bias=True)

        # Initialization
        torch.nn.init.xavier_uniform_(self.wo.weight)

    """ Forward """
    def forward(self,feature):
        zk = []
        for k in range(self.n_heads):
            zk.append( self.attns[k](feature) )
        z = torch.cat(zk,dim=-1)
        z = self.wo(z)
        z = F.dropout(z, p=self.dropout, training=self.training)
        return z


class FeedForward(nn.Module):
    """ Constructor """
    def __init__(self, n_features, n_heads, dff = 2048, dropout=0.1):
        super(FeedForward, self).__init__()
        # Parameters
        self.dmodel  = n_features
        self.n_heads = n_heads
        self.dv      = self.dmodel/n_heads
        self.dk      = self.dv
        self.dff     = dff
        self.dropout = dropout

        # Declare layers
        self. w1  = nn.Linear(self.dmodel,self.  dff ,bias=True)
        self. w2  = nn.Linear(self.  dff ,self.dmodel,bias=True)
        self.ReLU = nn.ReLU()

        # Initialization
        torch.nn.init.xavier_uniform_(self.w1.weight)
        torch.nn.init.xavier_uniform_(self.w2.weight)

    """ Forward """
    def forward(self,feature):
        x = self.ReLU(self.w1(feature))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.w2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class EncoderBlock(nn.Module):
    """ Constructor """
    def __init__(self, n_features,n_heads):
        super(EncoderBlock, self).__init__()
        # Modules
        self.MulHeadAttn = MultiHeadAttention(n_features,n_heads)
        self.FeedForward = FeedForward(n_features,n_heads,dff=2048)
        self.BatchNorm   = nn.BatchNorm1d(self.dmodel)

    """ Forward """
    def forward(self,x):
        x =   self.MulHeadAttn(x)
        x = x + self.BatchNorm(x)
        x =   self.FeedForward(x)
        x = x + self.BatchNorm(x)
        return x


""" ECA-Net Module
    --------------
    Ref: Wang, Qilong, et al. "Eca-net: Efficient channel attention for 
         deep convolutional neural networks." 
         arXiv preprint arXiv:1910.03151 (2019).
"""
class ECAnet(nn.Module):
    """ Constructor """
    def __init__(self, D, gamma=2,b=1):
        super(ECAnet, self).__init__()
        # Kernel size
        t = int( abs((math.log(D,2)+b)/gamma) )
        k = t if t%2 else t+1

        # Layers
        self.avgPool = nn.AdaptiveAvgPool1d(1)
        self.conv    = nn.Conv1d(1,1,kernel_size=k,padding=int(k/2),bias=False)

    """ Forward """
    def forward(self,x):        # [batch,L,D]
        x = x.transpose(1,2)    # [batch,D,L]
        y = self.avgPool(x)     # [batch,D,1]
        
        y = y.transpose(1,2)    # [batch,1,D]
        y = self.conv(y)        # [batch,1,D]
        
        return y
        
