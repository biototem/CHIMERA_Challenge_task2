"""
Graph Transformer Layer code mostly copy-paste but modified from
https://github.com/graphdeeplearning/graphtransformer/blob/main/layers/graph_transformer_layer.py
Transformer Layer code mostly copy-paste but modified from
https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py
"""

import dgl
import dgl.function as fn
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn


class T_MIL(nn.Module):
    def __init__(self, n_classes, architecture='LA_MIL', feat_dim=1024, latent_dim=512, num_heads=8, depth=2, max_limit_patch=None):

        """
        Args:
          num_classes: Number of targets to predict, note that we output the logits such that binary, multi-target binary and multi-class prediction is possible
          architecture: Architecture; choose between: LA_MIL, GA_MIL
          feat_dim: The output dimension of all tiles after feature extraction, depends on your feature extraction network
          latent_dim: Latent dimension for attention module
          num_heads: Number of heads for latent attention, commonly 8. 
          depth: number of attention layers

        """

        super().__init__()
        self.n_classes = n_classes
        self.architecture = architecture
        self.feat_dim = feat_dim
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.depth = depth
        self.max_limit_patch = max_limit_patch

        self.fc1 = nn.Linear(self.feat_dim, 512, bias=True)
        self.relu = nn.ReLU()


        # self.fc11 = nn.Linear(4, 16 , bias=True)
        # self.relu11 = nn.ReLU()


        # self.fc2 = nn.Linear(16, 2, bias=True)



        self.layers = nn.ModuleList([])

        # self.layers1 = nn.ModuleList([])
        # self.layers1.append(
        #     TransformerLayer(dim=16, heads=self.num_heads, use_ff=False, use_norm=True)
        # )
        if self.architecture == 'LA_MIL':
            for _ in range(self.depth):
                self.layers.append(
                    GraphTransformerLayer(in_dim=512, out_dim=self.latent_dim, num_heads=self.num_heads)
                )

        if self.architecture == 'GA_MIL':
            for _ in range(self.depth):
                self.layers.append(
                    TransformerLayer(dim=512, heads=self.num_heads, use_ff=False, use_norm=True)
                )

        # self.mlp_head = nn.Linear(self.latent_dim, self.n_classes, bias=True)
        self.mlp_head = nn.Linear(self.latent_dim, self.n_classes, bias=True)

    def forward(self, xs, coords=None, return_last_att=False, return_emb=False):
        '''

        :param xs: Tensor List [[N, C], [N2, C], ...]
        :param coords: Tensor List [[N, 2], [N2, 2], ...]
        :param return_last_att:
        :param return_emb:
        :return:
        '''

        if coords is None:
            coords = [None] * len(xs)

        out = []

        for x, cs in zip(xs, coords):
            x = x.cuda()
            cs = cs.cuda()
            if self.max_limit_patch is not None:
                if len(x) > self.max_limit_patch:
                    if self.training:
                        ids = torch.randperm(len(x))[:self.max_limit_patch]
                    else:
                        ids = torch.arange(len(x))[:self.max_limit_patch]
                    ids = ids.to(x.device)
                    x = x[ids]
                    if cs is not None:
                        cs = cs[ids]
            # x111 = x[:,768:770]
            # x222 = x[:,773:775]
            #
            # x1 = torch.cat((x111, x222), dim=1)
            # x = x[:,:768]

            # x1 = x1[None,]
            # x1 = self.fc11(x1)
            # x1 = self.relu11(x1)

            x = x[None,]
            x = self.fc1(x)
            x = self.relu(x)

            if self.architecture == 'LA_MIL':
                assert cs is not None
                knn1, knn2 = 16, 64  # adapt to your task
                g1, g2 = dgl.knn_graph(cs, knn1), dgl.knn_graph(cs, knn2)
                graphs = [g1, g2]
                for graph, layer in zip(graphs, self.layers):
                    x = layer(x, graph)

            if self.architecture == 'GA_MIL':
                for layer in self.layers:
                    x, att = layer(x)
                # for layer in self.layers1:
                #     x1, att = layer(x1)
            emb = x.mean(dim=1)

            # emb1 = x1.mean(dim=1)


            # emb2 = torch.cat((emb, emb1), dim=1)
            logits1 = self.mlp_head(emb)
            out.append(logits1)

            # return embedding
            # if return_emb:
            #     out.append(emb.detach())

            # return last attentions
            # if return_last_att and self.architecture == 'LA_MIL':
            #     out.append(graphs[-1].edata['score'])
            # elif return_last_att and self.architecture == 'GA_MIL':
            #     out.append(att)

        out = torch.cat(out, 0)

        return out


"""
    Graph Transformer
    
"""

"""
    Util functions
"""


def src_dot_dst(src_field, dst_field, out_field):
    def func(edges):
        return {out_field: (edges.src[src_field] * edges.dst[dst_field]).sum(-1, keepdim=True)}

    return func


def scaled_exp(field, scale_constant):
    def func(edges):
        # clamp for softmax numerical stability
        return {field: torch.exp((edges.data[field] / scale_constant).clamp(-5, 5))}

    return func


"""
    Single Attention Head
"""


class MultiHeadAttentionLayer(nn.Module):

    def __init__(self, in_dim, out_dim, num_heads, use_bias):
        super().__init__()

        self.out_dim = out_dim
        self.num_heads = num_heads

        if use_bias:
            self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            self.K = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            self.V = nn.Linear(in_dim, out_dim * num_heads, bias=True)
        else:
            self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=False)
            self.K = nn.Linear(in_dim, out_dim * num_heads, bias=False)
            self.V = nn.Linear(in_dim, out_dim * num_heads, bias=False)

    def propagate_attention(self, g):
        # Compute attention score

        g.apply_edges(src_dot_dst('K_h', 'Q_h', 'score'))  # , edges)

        g.apply_edges(scaled_exp('score', np.sqrt(self.out_dim)))

        # Send weighted values to target nodes
        eids = g.edges()

        g.send_and_recv(eids, fn.copy_e('score', 'score'), fn.sum('score', 'att'))

        g.send_and_recv(eids, fn.u_mul_e('V_h', 'score', 'V_h'), fn.sum('V_h', 'wV'))

        g.send_and_recv(eids, fn.copy_e('score', 'score'), fn.sum('score', 'z'))

    def forward(self, h, g):

        Q_h = self.Q(h)
        K_h = self.K(h)
        V_h = self.V(h)

        # Reshaping into [num_nodes, num_heads, feat_dim] to 
        # get projections for multi-head attention

        g.ndata['Q_h'] = Q_h.view(-1, self.num_heads, self.out_dim)
        g.ndata['K_h'] = K_h.view(-1, self.num_heads, self.out_dim)
        g.ndata['V_h'] = V_h.view(-1, self.num_heads, self.out_dim)

        self.propagate_attention(g)

        head_out = g.ndata['wV'] / g.ndata['z']

        return head_out


class GraphTransformerLayer(nn.Module):
    """
        Param: 
    """

    def __init__(self, in_dim, out_dim, num_heads, dropout=0.0, layer_norm=False, residual=True, use_bias=False):
        super().__init__()

        self.in_channels = in_dim
        self.out_channels = out_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.residual = residual
        self.layer_norm = layer_norm

        self.attention = MultiHeadAttentionLayer(in_dim, out_dim // num_heads, num_heads, use_bias)

        self.O = nn.Linear(out_dim, out_dim)

        if self.layer_norm:
            self.layer_norm1 = nn.LayerNorm(out_dim)

        if self.layer_norm:
            self.layer_norm2 = nn.LayerNorm(out_dim)

    def forward(self, h, g):
        h_in1 = h  # for first residual connection

        # multi-head attention out
        attn_out = self.attention(h, g)

        h = attn_out.view(-1, self.out_channels)

        h = F.dropout(h, self.dropout, training=self.training)

        h = self.O(h)

        if self.residual:
            h = h_in1 + h  # residual connection

        if self.layer_norm:
            h = self.layer_norm1(h)

        return h

    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, heads={}, residual={})'.format(self.__class__.__name__,
                                                                                   self.in_channels,
                                                                                   self.out_channels, self.num_heads, self.residual)


"""
    Transformer  
"""


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class Attention(nn.Module):
    def __init__(self, dim=512, heads=8, dim_head=512 // 8, dropout=0.1):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        # dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        # attn = self.attend(dots)
        # out = torch.matmul(attn, v)
        # out = rearrange(out, 'b h n d -> b n (h d)')

        dots = F.scaled_dot_product_attention(q, k, v,scale=self.scale)
        out2 = rearrange(dots, 'b h n d -> b n (h d)')


        return self.to_out(out2), None


class FeedForward(nn.Module):
    def __init__(self, dim=512, hidden_dim=1024, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class TransformerLayer(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm, dim=512, heads=8, use_ff=False, use_norm=True):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = Attention(dim=dim, heads=heads, dim_head=dim // heads)
        self.use_ff = use_ff
        self.use_norm = use_norm
        if self.use_ff:
            self.ff = FeedForward(dim=dim)

    def forward(self, x):

        shortcut = x

        x, att = self.attn(x)
        x = shortcut + x

        if self.use_ff:
            x = self.ff(x) + x

        return x, att


if __name__ == '__main__':
    import torch, dgl, random

    # data
    batch_size = 1  # num of WSIs=1
    num_tiles = 6000  # num of used tiles from WSI
    feat_dim = 768  # or feature dimension, dependent on feature extractor
    num_classes = 2  # num of targets to predict
    tile_coords = torch.tensor([(random.random(), random.random()) for _ in range(num_tiles)])  # tile coordinates from WSI
    wsi = torch.randn(batch_size, num_tiles, feat_dim)

    # model
    m = T_MIL(
        n_classes=num_classes,
        architecture='LA_MIL',
        feat_dim=feat_dim,
        latent_dim=512,
        num_heads=8,
        depth=2,
        max_limit_patch=3000,
    )
    wsi = wsi.cuda(0)
    tile_coords = tile_coords.cuda(0)
    m.cuda(0)

    logits = m(wsi, [tile_coords]*len(wsi))

    # Multi Target (e.g, predict if multiple mutations are present: TP53 True, BRAF True, MSI False, TMB True <-> [1, 1, 0, 1]
    multi_target_binaries = torch.where(torch.sigmoid(logits[0]) > 0.5, 1., 0.)
    logits.mean().backward()
    print(1)

    # LAMIL bs=1显存占用
    # 2048 2.4g
    # 13000 8.9g
