import torch
from   torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange

# SOURCE: https://github.com/lucidrains/lambda-networks/blob/main/lambda_networks/lambda_networks.py

# helpers functions
def exists(val):
    return val is not None
def default(val, d):
    return val if exists(val) else d
def calc_rel_pos(n):
    pos = torch.meshgrid(torch.arange(n), torch.arange(n))
    pos = rearrange(torch.stack(pos), 'n i j -> (i j) n')  # [n*n, 2] pos[n] = (i, j)
    rel_pos = pos[None, :] - pos[:, None]                  # [n*n, n*n, 2] rel_pos[n, m] = (rel_i, rel_j)
    rel_pos += n - 1                                       # shift value range from [-n+1, n-1] to [0, 2n-2]
    return rel_pos

"""
class λLayer(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out = None,
        dim_k   =   16,
        n = None,
        r = None,
        heads = 4,
        dim_u = 1):
        super().__init__()
        dim_out = default(dim_out, dim_in)
        self.u = dim_u # intra-depth dimension
        self.heads = heads

        assert (dim_out % heads) == 0, 'values dimension must be divisible by number of heads for multi-head query'
        dim_v = dim_out // heads

        self.to_q = nn.Conv2d(dim_in, dim_k * heads, 1, bias = False)
        self.to_k = nn.Conv2d(dim_in, dim_k * dim_u, 1, bias = False)
        self.to_v = nn.Conv2d(dim_in, dim_v * dim_u, 1, bias = False)

        self.norm_q = nn.BatchNorm2d(dim_k * heads)
        self.norm_v = nn.BatchNorm2d(dim_v * dim_u)


        # SOURCE: https://github.com/leaderj1001/LambdaNetworks/blob/main/model.py
        # -----------------------------------------------------------------------------------------
        # self.kk    = k     = 16
        # self.uu    = u     =  1
        # self.vv    = out_channels // heads
        # self.mm    = m     = 23
        # self.heads = heads =  4
        
        # if self.local_context:
        #     self.embedding = nn.Parameter(torch.randn([self.kk, self.uu, 1, m, m]), requires_grad=True)
        # else:
        #     self.embedding = nn.Parameter(torch.randn([self.kk, self.uu]), requires_grad=True)
        # 
        self.embedding = nn.Parameter(torch.randn([dim_k, dim_u]), requires_grad=True)

        # ORGINAL
        # -----------------------------------------------------------------------------------------
        # self.local_contexts = exists(r)
        # if exists(r):
        #     assert (r % 2) == 1, 'Receptive kernel size should be odd'
        #     self.pos_conv = nn.Conv3d(dim_u, dim_k, (1, r, r), padding = (0, r // 2, r // 2))
        # else:
        #     assert exists(n), 'You must specify the window size (n=h=w)'
        #     rel_lengths = 2 * n - 1
        #     self.rel_pos_emb = nn.Parameter(torch.randn(rel_lengths, rel_lengths, dim_k, dim_u))
        #     self.rel_pos = calc_rel_pos(n)
        # 

    def forward(self, x):
        b, c, hh, ww, u, h = *x.shape, self.u, self.heads

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        q = self.norm_q(q)
        v = self.norm_v(v)

        q = rearrange(q, 'b (h k) hh ww -> b h k (hh ww)', h = h)
        k = rearrange(k, 'b (u k) hh ww -> b u k (hh ww)', u = u)
        v = rearrange(v, 'b (u v) hh ww -> b u v (hh ww)', u = u)

        k = k.softmax(dim=-1)

        λc = einsum('b u k m, b u v m -> b k v', k, v)
        Yc = einsum('b h k n, b k v -> b h v n', q, λc)

        # if self.local_contexts:
        #     v = rearrange(v, 'b u v (hh ww) -> b u v hh ww', hh = hh, ww = ww)
        #     λp = self.pos_conv(v)
        #     Yp = einsum('b h k n, b k v n -> b h v n', q, λp.flatten(3))
        # else:
        #     n, m = self.rel_pos.unbind(dim = -1)
        #     rel_pos_emb = self.rel_pos_emb[n, m]
        #     λp = einsum('n m k u, b u v m -> b n k v', rel_pos_emb, v)
        #     Yp = einsum('b h k n, b n k v -> b h v n', q, λp)
        
        # embedding: torch.Size([16, 1])
        # v        : torch.Size([ 1, 1, 16, 288])
        # v ->   b, v, uu, w * h
        λp = torch.einsum('ku,buvn->bkvn', self.embedding, v)
        Yp = torch.einsum('bhkn,bkvn->bhvn', q, λp)

        Y = Yc + Yp
        out = rearrange(Y, 'b h v (hh ww) -> b (h v) hh ww', hh = hh, ww = ww)
        return out
"""

"""
dim_in,
dim_out = None,
dim_k   =   16,
n = None,
r = None,
heads = 4,
dim_u = 1):
"""
class λLayer(nn.Module):
    def __init__(self, in_channels, out_channels, heads=4, k=16, u=1, m=23):
        super(λLayer, self).__init__()
        self.kk, self.uu, self.vv, self.mm, self.heads = k, u, out_channels // heads, m, heads
        self.local_context = True if m > 0 else False
        self.padding = (m - 1) // 2

        self.queries = nn.Sequential(
            nn.Conv2d(in_channels, k * heads, kernel_size=1, bias=False),
            nn.BatchNorm2d(k * heads)
        )
        self.keys = nn.Sequential(
            nn.Conv2d(in_channels, k * u, kernel_size=1, bias=False),
        )
        self.values = nn.Sequential(
            nn.Conv2d(in_channels, self.vv * u, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.vv * u)
        )

        self.softmax = nn.Softmax(dim=-1)

        if self.local_context:
            self.embedding = nn.Parameter(torch.randn([self.kk, self.uu, 1, m, m]), requires_grad=True)
        else:
            self.embedding = nn.Parameter(torch.randn([self.kk, self.uu]), requires_grad=True)

    def forward(self, x):
        n_batch, C, w, h = x.size()

        queries = self.queries(x).view(n_batch, self.heads, self.kk, w * h) # b, heads, k // heads, w * h
        softmax = self.softmax(self.keys(x).view(n_batch, self.kk, self.uu, w * h)) # b, k, uu, w * h
        values = self.values(x).view(n_batch, self.vv, self.uu, w * h) # b, v, uu, w * h

        lambda_c = torch.einsum('bkum,bvum->bkv', softmax, values)
        y_c = torch.einsum('bhkn,bkv->bhvn', queries, lambda_c)

        if self.local_context:
            values = values.view(n_batch, self.uu, -1, w, h)
            lambda_p = F.conv3d(values, self.embedding, padding=(0, self.padding, self.padding))
            lambda_p = lambda_p.view(n_batch, self.kk, self.vv, w * h)
            y_p = torch.einsum('bhkn,bkvn->bhvn', queries, lambda_p)
        else:
            lambda_p = torch.einsum('ku,bvun->bkvn', self.embedding, values)
            y_p = torch.einsum('bhkn,bkvn->bhvn', queries, lambda_p)

        out = y_c + y_p
        out = out.contiguous().view(n_batch, -1, w, h)

        return out
        
