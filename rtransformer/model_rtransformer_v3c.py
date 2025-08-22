'''
聚合模型V3
改自rtv2，预计只改动聚合方法

不管了，直接来一个大桶注意力解决一切，草，又不是干不了
'''
import math
import sys
import os
# sys.path.append(os.path.dirname(__file__))

import torch
import torch.nn as nn
import torch.nn.functional as F
from model_utils_torch import MultiHeadAttention
from model_utils_torch import FlashQuadAttention
# from model_utils_torch.rev.rev_utils import rev_sequential_backward_wrapper
# from model_utils_torch.rev.rev_blocks import RevSequential, RevGroupBlock
from torch.utils.checkpoint import checkpoint, checkpoint_sequential


def ConvBnAct(in_ch, out_ch, ker_sz, stride, pad, act=nn.Identity(), group=1, dilation=1):
    return nn.Sequential(nn.Conv2d(in_ch, out_ch, ker_sz, stride, pad, groups=group, bias=False, dilation=dilation),
                         nn.BatchNorm2d(out_ch, eps=1e-8, momentum=0.1),
                         act)


# def bucket_shuffle(s, bucket_size, base):
#     '''
#     桶混洗
#     :param s: shape [1, L, C]
#     :return:
#     '''
#     n_sub = s.shape[1] // base
#     group_s = s.reshape(n_bucket, shuffle_group_size, bucket_size // shuffle_group_size, s.shape[1])
#     group_s = group_s.permute(0, 2, 1, 3)
#     group_s = group_s.reshape(1, n_bucket, bucket_size, s.shape[1])



def ortho_pad_1d(s, base):
    '''
    正交填充
    :param s:
    :param dim:
    :param base:
    :return:
    '''
    n_pad = base - s.shape[0] % base
    n_ch = s.shape[1]
    pad_s = torch.zeros([n_pad, n_ch], dtype=s.dtype, device=s.device)
    pad_ids1 = torch.arange(n_pad, device=s.device)
    pad_ids2 = pad_ids1 % n_ch
    pad_s[pad_ids1, pad_ids2] = 1.
    s = torch.cat([s, pad_s], 0)
    return s


def split_group_v2(s: torch.Tensor, bucket_size, shuffle_group_size):
    assert s.shape[0] == 1

    # s.shape [B, L, C]
    s = s[0]
    # s.shape[L, C]
    with torch.no_grad():
        multi_groups_ids = [list(range(s.shape[0]))]
        lens = torch.linalg.norm(s, ord=2, dim=1)

        w1 = torch.linspace(0, 1, s.shape[0], dtype=s.dtype, device=s.device)

        v1_id = torch.argmax(lens)

        weight = torch.cosine_similarity(s[v1_id:v1_id+1], s, 1)

        v2_id = torch.argmax(weight)

        weight += torch.cosine_similarity(s[v2_id:v2_id+1], s, 1)

        sorted_ids = torch.argsort(weight, 0)

    s = s[sorted_ids]
    s = ortho_pad_1d(s, bucket_size)
    # s [L, C]
    n_bucket = s.shape[0] // bucket_size
    group_s = s.reshape(n_bucket, shuffle_group_size, bucket_size // shuffle_group_size, s.shape[1])
    group_s = group_s.permute(0, 2, 1, 3)
    group_s = group_s.reshape(1, n_bucket, bucket_size, s.shape[1])
    # gs [B, G, L2, C]
    return group_s


def split_group_v3(s: torch.Tensor, bucket_size, shuffle_group_size):
    assert s.shape[0] == 1

    # s.shape [B, L, C]
    s = s[0]
    # s.shape[L, C]
    with torch.no_grad():
        multi_groups_ids = [list(range(s.shape[0]))]
        lens = torch.linalg.norm(s, ord=2, dim=1)

        w = torch.zeros(s.shape[0], dtype=s.dtype, device=s.device)
        w_space = torch.linspace(0, 1, s.shape[0], dtype=s.dtype, device=s.device).square()

        #
        v1_id = torch.argmax(lens)
        cw = torch.cosine_similarity(s[v1_id], s, 1)
        w += w_space[torch.argsort(cw)]

        # -
        v2_id = torch.argmin(w)
        cw = torch.cosine_similarity(s[v2_id], s, 1)
        w += w_space[torch.argsort(cw)]

        v2_id = torch.argmin(w)
        cw = torch.cosine_similarity(s[v2_id], s, 1)
        w += w_space[torch.argsort(cw)]

        v2_id = torch.argmin(w)
        cw = torch.cosine_similarity(s[v2_id], s, 1)
        w += w_space[torch.argsort(cw)]

        sorted_ids = torch.argsort(w, 0)

    s = s[sorted_ids]
    s = ortho_pad_1d(s, bucket_size)
    # s [L, C]
    n_bucket = s.shape[0] // bucket_size
    group_s = s.reshape(1, n_bucket, bucket_size, s.shape[1])
    # gs [B, G, L2, C]
    return group_s


def split_group_v4(s: torch.Tensor, bucket_size, shuffle_group_size):
    assert s.shape[0] == 1

    # s.shape [B, L, C]
    s = s[0]
    # s.shape[L, C]
    with torch.no_grad():
        multi_groups_ids = [list(range(s.shape[0]))]
        lens = torch.linalg.norm(s, ord=2, dim=1)

        #
        v1_id = torch.argmax(lens)
        cw = torch.cosine_similarity(s[v1_id], s, 1)

        sorted_ids = torch.argsort(cw, 0, descending=True)

    s = s[sorted_ids]
    s = ortho_pad_1d(s, bucket_size)
    # s [L, C]
    n_bucket = s.shape[0] // bucket_size
    group_s = s.reshape(1, n_bucket, bucket_size, s.shape[1])
    # gs [B, G, L2, C]
    return group_s


def make_group_batch_v2(group_s, batch_group_size):
    batch_group = []
    for b_i in range(int(math.ceil(group_s.shape[1] / batch_group_size))):
        g = group_s[:, b_i*batch_group_size: (b_i+1)*batch_group_size]
        batch_group.append(g)
    return batch_group


# class Block(nn.Module):
# class Block(torch.jit.ScriptModule):
#     def __init__(self, in_dim=256, out_dim=256, head_dim=32, n_head=8):
#         super().__init__()
#         assert in_dim == out_dim
#         self.norm = nn.LayerNorm(out_dim, eps=1e-8)
#         self.att1 = MultiHeadAttention(in_dim, out_dim, head_dim, n_head)
#         self.mlp = nn.Sequential(nn.Linear(out_dim, out_dim*2),
#                                  nn.GELU(),
#                                  nn.Linear(out_dim*2, out_dim),
#                                  )
#
#     @torch.jit.script_method
#     def func(self, x):
#         y = x
#         y = self.norm(y)
#         y = self.att1(y)
#         y = self.mlp(y)
#         y = x + y
#         return y
#
#     @torch.jit.script_method
#     def forward(self, x):
#         y = self.func(x)
#         return y


class StdClip(nn.Module):
    def __init__(self, std, dim):
        super().__init__()
        self.std = std
        self.dim = dim

    def forward(self, x):
        ori_std = x.std(self.dim, unbiased=False, keepdim=True).clamp_min(1e-4)

        factor = self.std / ori_std
        # factor = factor.clamp(0.01, 100.)
        factor = factor.clamp(0.1, 10.)

        y = x * factor
        return y


class RmsClip(nn.Module):
    def __init__(self, rms, dim):
        super().__init__()
        self.rms = rms
        self.dim = dim

    def forward(self, x):
        ori_rms = x.square().mean(self.dim, keepdim=True).__add__(1e-4).sqrt()

        factor = self.rms / ori_rms
        # factor = factor.clamp(0.01, 100.)
        factor = factor.clamp(0.1, 10.)

        y = x * factor
        return y


class TransLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.den1 = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        # self.norm = StdClip(1., [1, 2])
        # self.norm = RmsClip(2., [1, 2])
        self.act = nn.GELU()

    def forward(self, x):
        y = self.den1(x)
        y = self.norm(y)
        y = self.act(y)
        return y


class Block(torch.jit.ScriptModule):
    def __init__(self, in_dim=256, expand_dim=768, squeeze_dim=512):
        super().__init__()
        # assert in_dim == out_dim
        self.attn = FlashQuadAttention(in_dim, in_dim, expand_dim, squeeze_dim, use_norm=True, use_skip=True)

    @torch.jit.script_method
    def forward(self, x):
        y = self.attn(x)
        return y


class SubNet(nn.Module):
    def __init__(self, in_dim, inter_dim, out_dim, n_block, expand_dim, squeeze_dim):
        super().__init__()
        assert inter_dim == in_dim == out_dim

        ms = []
        # ms.extend([
        #     TransLayer(in_dim, inter_dim),
        # ])
        for mi in range(n_block):
            m = Block(inter_dim, expand_dim, squeeze_dim)
            ms.append(m)
        # ms.extend([
        #     TransLayer(inter_dim, out_dim),
        # ])
        self.ms = nn.Sequential(*ms)

    def block_warp(self, x):
        '''
        检查点方法，节省显存，慢。
        显存足够时用这个
        :param x:
        :return:
        '''
        if self.training and True:
            y = checkpoint_sequential(self.ms, len(self.ms), x, preserve_rng_state=False)
        else:
            y = self.ms(x)
        return y

    def forward(self, x):
        y = self.block_warp(x)
        return y


class Net(nn.Module):
    model_id = 'rtv3'

    def __init__(self, in_dim, out_dim, inter_dim=512, n_block=6, expand_dim=768, squeeze_dim=256, bucket_size=64, bucket_batch_num=512, shuffle_group_size=32):
        super().__init__()
        # 每个桶的最大长度，用于控制注意力矩阵的大小，会影响结果和速度
        self.bucket_size = bucket_size
        # 每批次送多少个桶进入网络，不影响结果，只影响速度
        self.bucket_batch_num = bucket_batch_num
        # 每组混淆大小
        self.shuffle_group_size = shuffle_group_size

        # self.in1 = nn.Sequential(nn.Linear(in_dim, inter_dim, True),
        #                          StdFix(0.7, dim=[2]),
        #                          nn.GELU())

        self.in1 = nn.Sequential(nn.Linear(in_dim, inter_dim),
                                 # nn.LayerNorm(inter_dim),
                                 )

        self.net = SubNet(inter_dim, inter_dim, inter_dim, n_block, expand_dim, squeeze_dim)

        self.tr_layer = nn.Sequential(nn.LayerNorm(inter_dim),
                                      nn.LeakyReLU(inplace=True),
                                      nn.Linear(inter_dim, inter_dim),
                                      )

        # self.tr_layer = TransLayer(inter_dim, inter_dim)

        self.out1 = nn.Sequential(
            nn.LayerNorm(inter_dim),
            nn.LeakyReLU(inplace=True),
            # nn.Linear(inter_dim, inter_dim),
            nn.Linear(inter_dim, out_dim)
        )

    def forward(self, xs):
        os = []
        for x in xs:
            x = x[None,]
            x = self.in1(x)
            ys = []
            while True:
                gys = []
                gxs = split_group_v4(x, self.bucket_size, self.shuffle_group_size)
                gxs = make_group_batch_v2(gxs, self.bucket_batch_num)
                for gx in gxs:
                    gy = self.net(gx)
                    gy = gy.mean(2, keepdim=False)
                    gys.append(gy)
                gys = torch.concat(gys, 1)
                # [B, L2, C]
                ys.append(self.tr_layer(gys))
                if gys.shape[1] == 1:
                    break
                x = gys
            ys = torch.cat(ys, 1)
            y = ys.mean(dim=1)
            os.append(y)

        y = torch.cat(os, 0)
        y = self.out1(y)
        return y


if __name__ == '__main__':
    import model_utils_torch

    device = torch.device('cuda:0')
    a = torch.randn(5, 30000, 1024, device=device)

    net = Net(1024, 128, inter_dim=1024, n_block=4, expand_dim=128, squeeze_dim=64, bucket_size=256, bucket_batch_num=512).to(device)
    model_utils_torch.print_params_size(net)

    y = net(a)
    y.abs().sum().backward()
    print(y.shape)

    del a
    torch.cuda.empty_cache()

    optim = torch.optim.AdamW(net.parameters(), 1e-4)

    for it in range(1000):

        x = torch.randn([30, 3000, 1024], device=device)
        y = torch.full([30], fill_value=0, dtype=torch.long, device=device)
        x[:15, :20, :128] = torch.rand_like(x[:15, :20, :128], device=device) * 1.2 - 1
        y[:15] = 0
        x[15:, :20, :128] = torch.rand_like(x[15:, :20, :128], device=device) * -1.2 + 1
        y[15:] = 1

        optim.zero_grad()
        o = net(x)
        # loss = o.abs().sum()
        loss = F.cross_entropy(o, y, label_smoothing=0.2)
        acc = torch.mean(torch.argmax(o, 1) == y, dtype=torch.float32).item()
        loss.backward()
        print('it', it, 'acc', acc, 'loss', loss.item())
        optim.step()
