import math 
import torch
import torch.nn as nn
import torch.nn.functional as F

# Transformer
# http://jalammar.github.io/illustrated-transformer/
# https://arxiv.org/pdf/1706.03762.pdf
# https://towardsdatascience.com/how-to-code-the-transformer-in-pytorch-24db27c8f9ec
# https://ufal.mff.cuni.cz/pbml/110/art-popel-bojar.pdf
class SelfAttention(nn.Module):
    """ Constructor """
    def __init__(self, n_features,n_heads):
        super(SelfAttention, self).__init__()
        # Parameters
        self.dmodel  = n_features
        self.n_heads = n_heads
        self.dv = self.dmodel/n_heads
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


class Transformer(nn.Module):
    """ Constructor """
    def __init__(self, n_features,n_heads):
        super(Transformer, self).__init__()
        
    

    """ Forward """
    def forward(self,feature):
        return None

