'''
聚合模型

不管了，直接来一个大桶注意力解决一切，草，又不是干不了
'''
import math
import sys
import os
sys.path.append(os.path.dirname(__file__))

import torch
import torch.nn as nn
import torch.nn.functional as F
from model_utils_torch import MultiHeadAttention
from model_utils_torch.rev.rev_utils import rev_sequential_backward_wrapper
from model_utils_torch.rev.rev_blocks import RevSequential, RevGroupBlock
from torch.utils.checkpoint import checkpoint, checkpoint_sequential



def ConvBnAct(in_ch, out_ch, ker_sz, stride, pad, act=nn.Identity(), group=1, dilation=1):
    return nn.Sequential(nn.Conv2d(in_ch, out_ch, ker_sz, stride, pad, groups=group, bias=False, dilation=dilation),
                         nn.BatchNorm2d(out_ch, eps=1e-8, momentum=0.9),
                         act)


def split_group(s: torch.Tensor, group_size):
    assert s.shape[0] == 1
    with torch.no_grad():
        s_len = torch.linalg.norm(s, ord=2, dim=2)[0]
        # s_len = s.mean(dim=2)[0]
        ids = torch.argsort(s_len)
    s = s[:, ids, :]
    group_s = torch.split(s, group_size, dim=1)
    return group_s


def make_group_batch(group_s, batch_group_size):
    batch_group = [group_s[-1]]
    group_s = group_s[:-1]
    for b_i in range(0, int(math.ceil(len(group_s) / batch_group_size))):
        g = group_s[b_i*batch_group_size: (b_i+1)*batch_group_size]
        g = torch.cat(g, 0)
        batch_group.append(g)
    return batch_group


# # class Block(nn.Module):
# class Block(torch.jit.ScriptModule):
#     def __init__(self, in_dim=256, out_dim=256):
#         super().__init__()
#         assert in_dim == out_dim
#
#         self.att1 = MultiHeadAttention(in_dim, out_dim, 32, 8)
#         self.mlp = nn.Sequential(nn.Linear(out_dim, out_dim*2),
#                                  nn.GELU(),
#                                  nn.Linear(out_dim*2, out_dim),
#                                  )
#         self.a = nn.Parameter(torch.zeros(2))
#
#     @torch.jit.script_method
#     def func(self, x):
#         y = x
#         y = self.att1(y) * self.a[0]
#         y = self.mlp(y) * self.a[1]
#         return y
#
#     @torch.jit.script_method
#     def forward(self, x1, x2):
#         y1 = x2
#         y2 = x1 + self.func(x2)
#         return y1, y2
#
#     @torch.jit.script_method
#     def invert(self, y1, y2):
#         x2, y = y1, y2
#         x1 = y - self.func(x2)
#         return x1, x2


# class SubNet(nn.Module):
#     def __init__(self, in_dim, inter_dim, out_dim, head_dim=8, n_head=4):
#         super().__init__()
#         self.in1 = nn.Linear(in_dim, inter_dim)
#         ms = []
#         for mi in range(6):
#             m = Block(inter_dim, inter_dim)
#             ms.append(m)
#         self.ms = RevSequential(ms)
#         self.out1 = nn.Linear(inter_dim, out_dim)
#
#     # def rev_warp(self, x):
#     #     '''
#     #     可逆方法，更加节省显存，更慢
#     #     显存足够时不建议使用
#     #     :param x:
#     #     :return:
#     #     '''
#     #     x = x * 0.5
#     #     if self.training and True:
#     #         y1, y2 = rev_sequential_backward_wrapper(self.ms, x, x, preserve_rng_state=False)
#     #     else:
#     #         y1, y2 = self.ms(x, x)
#     #     y = y1 + y2
#     #     return y
#
#     def rev_warp(self, x):
#         '''
#         检查点方法，节省显存，慢。
#         显存足够时用这个
#         :param x:
#         :return:
#         '''
#         x = x * 0.5
#         if self.training and True:
#             y1, y2 = checkpoint(self.ms, x, x, preserve_rng_state=False)
#         else:
#             y1, y2 = self.ms(x, x)
#         y = y1 + y2
#         return y
#
#     def forward(self, x):
#         y = self.in1(x)
#         y = self.rev_warp(y)
#         y = self.out1(y)
#         return y


# class Block(nn.Module):
class Block(torch.jit.ScriptModule):
    def __init__(self, in_dim=256, out_dim=256, head_dim=32, n_head=8):
        super().__init__()
        assert in_dim == out_dim
        self.norm = nn.LayerNorm(out_dim, eps=1e-8)
        self.att1 = MultiHeadAttention(in_dim, out_dim, head_dim, n_head)
        self.mlp = nn.Sequential(nn.Linear(out_dim, out_dim*2),
                                 nn.GELU(),
                                 nn.Linear(out_dim*2, out_dim),
                                 )
        # self.a = nn.Parameter(torch.zeros(1))

    @torch.jit.script_method
    def func(self, x):
        y = x
        y = self.norm(y)
        y = self.att1(y)
        y = self.mlp(y)
        y = x + y
        return y

    @torch.jit.script_method
    def forward(self, x):
        y = self.func(x)
        return y


class SubNet(nn.Module):
    def __init__(self, in_dim, inter_dim, out_dim, n_block=6, head_dim=32, n_head=8):
        super().__init__()
        self.in1 = nn.Linear(in_dim, inter_dim)
        ms = []
        for mi in range(n_block):
            m = Block(inter_dim, inter_dim, head_dim, n_head)
            ms.append(m)
        self.ms = nn.Sequential(*ms)
        self.out1 = nn.Linear(inter_dim, out_dim)

    def block_warp(self, x):
        '''
        检查点方法，节省显存，慢。
        显存足够时用这个
        :param x:
        :return:
        '''
        if self.training and True:
            y = checkpoint(self.ms, x, preserve_rng_state=False)
        else:
            y = self.ms(x)
        return y

    def forward(self, x):
        y = self.in1(x)
        y = self.block_warp(y)
        y = self.out1(y)
        return y


class Net(nn.Module):
    model_id = 'net2'

    def __init__(self, in_dim, out_dim, inter_dim=256, group_size=256, group_batch_size=512):
        super().__init__()
        # 每个桶的最大长度，用于控制注意力矩阵的大小
        self.group_size = group_size
        # 每批次送多少个桶进入网络
        self.group_batch_size = group_batch_size
        self.in1 = nn.Sequential(nn.Linear(in_dim, inter_dim, True))
        self.net = SubNet(inter_dim, inter_dim, inter_dim, 8, 32, 8)
        self.out1 = nn.Sequential(#nn.Linear(inter_dim, inter_dim, True),
                                  #nn.LayerNorm(inter_dim),
                                  #nn.GELU(),
                                  nn.Linear(inter_dim, out_dim, True))

    def forward(self, xs):
        os = []
        for x in xs:
            x = x[None,]
            x = self.in1(x)
            ys = []
            while True:
                gys = []
                gxs = split_group(x, self.group_size)
                gxs = make_group_batch(gxs, self.group_batch_size)
                for gx in gxs:
                    gy = self.net(gx)
                    gy = gy.mean(1, keepdim=True)
                    gys.append(gy)
                gys = torch.concat(gys, 0)
                gys = torch.transpose(gys, 0, 1)
                ys.append(gys.mean(dim=1, keepdim=True))
                # ys.append(gys)
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

    a = torch.randn(5, 80000, 2048).cuda(0)

    net = Net(2048, 128, inter_dim=512, group_size=256, group_batch_size=8).cuda(0)
    model_utils_torch.print_params_size(net)

    y = net(a)
    y.abs().sum().backward()
    print(y.shape)

    optim = torch.optim.Adam(net.parameters(), 1e-4)
    for _ in range(1000):
        optim.zero_grad()
        y = net(a)
        loss = y.abs().sum()
        loss.backward()
        print(loss.item())
        optim.step()

#
#
# if __name__ == '__main__':
#     import model_utils_torch
#
#     device = torch.device('cuda:0')
#     a = torch.randn(5, 30000, 1024, device=device)
#
#     net = Net(1024, 128, inter_dim=512, group_size=256).cuda(0)
#     model_utils_torch.print_params_size(net)
#
#     # y = net(a)
#     # y.abs().sum().backward()
#     # print(y.shape)
#
#     del a
#     torch.cuda.empty_cache()
#
#     optim = torch.optim.Adam(net.parameters(), 1e-4)
#
#     for it in range(1000):
#
#         x = torch.randn([20, 3000, 1024], device=device)
#         y = torch.full([20], fill_value=0, dtype=torch.long, device=device)
#         x[:10, :20, :128] = torch.rand_like(x[:10, :20, :128], device=device) * 1.2 - 1
#         y[:10] = 0
#         x[10:, :20, :128] = torch.rand_like(x[10:, :20, :128], device=device) * -1.2 + 1
#         y[10:] = 1
#
#         optim.zero_grad()
#         o = net(x)
#         # loss = o.abs().sum()
#         loss = F.cross_entropy(o, y, label_smoothing=0.2)
#         acc = torch.mean(torch.argmax(o, 1) == y, dtype=torch.float32).item()
#         loss.backward()
#         print('it', it, 'acc', acc, 'loss', loss.item())
#         optim.step()
