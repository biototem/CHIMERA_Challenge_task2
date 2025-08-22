import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from force_relative_import import enable_force_relative_import


with enable_force_relative_import():
    from .model.network import Classifier_1fc
    from .model.network import DimReduction
    from .model.Attention import Attention_Gated, Attention_with_Classifier


def get_cam_1d(w, features):
    # tweight = list(classifier.parameters())[-2]
    cam_maps = torch.einsum('bgf,cf->bcg', [features, w])
    return cam_maps
from torch.nn import BatchNorm1d, Sequential, Linear, ReLU, Tanh, LeakyReLU, ELU, SELU, GELU, Sigmoid, Dropout

class DTFD_MIL(nn.Module):
    def __init__(self, feat_dim, n_cls=2, mDim=512, numLayer_Res=0, distill='MaxMinS', numGroup=1, instance_per_group=1):
        super().__init__()
        assert distill in ('MaxMinS', 'MaxS', 'AFS')
        self.distill = distill
        self.numGroup = numGroup
        self.instance_per_group = instance_per_group

        self.classifier = Classifier_1fc(mDim, n_cls, droprate=0)
        self.attention = Attention_Gated(mDim)
        self.dimReduction = DimReduction(feat_dim, mDim, numLayer_Res=numLayer_Res)
        self.attCls = Attention_with_Classifier(L=mDim, num_cls=n_cls, droprate=0)
        self.mDim = mDim
    def single_forward(self, x):
        ## x [B, L, C]
        # x [L, C]

        distill = self.distill
        numGroup = self.numGroup
        instance_per_group = self.instance_per_group

        if self.training:
            ids = torch.randperm(x.shape[0], device=x.device)
        else:
            ids = torch.arange(x.shape[0], device=x.device)

        slide_pseudo_feat = []
        slide_sub_preds = []
        # slide_sub_labels = []

        group_size = int(math.ceil(x.shape[0] / numGroup))
        group_ids = torch.split(ids, group_size, 0)

        for each_group_ids in group_ids:
            # slide_sub_labels.append(label)

            sub_feat = x[each_group_ids]
            mid_feat = self.dimReduction(sub_feat)
            AA = self.attention(mid_feat).squeeze(0)
            att_feats = torch.einsum('ns,n->ns', mid_feat, AA)  ### n x fs
            att_feat_tensor = torch.sum(att_feats, dim=0).unsqueeze(0)  ## 1 x fs

            # clinic_batch_8 = [[1, 2, 3, 4, 5, 6, 7, 8] for iu in range(1)]
            # clinic_batch_8 = torch.from_numpy(np.array(clinic_batch_8)).float()
            # out11 = torch.cat((att_feat_tensor, clinic_batch_8), 1)
            # 新加
            # fc = Sequential(
            #     Linear(self.mDim + 8, self.mDim + 8),
            #     ELU(),
            #     Linear(self.mDim + 8, self.mDim),
            # )
            # new_out11 =fc(out11)
            # print(new_out11.shape)
            predict = self.classifier(att_feat_tensor)  ### 1 x 2


            slide_sub_preds.append(predict)

            patch_pred_logits = get_cam_1d(self.classifier.fc.weight, att_feats.unsqueeze(0)).squeeze(0)  ###  cls x n
            patch_pred_logits = torch.transpose(patch_pred_logits, 0, 1)  ## n x cls
            patch_pred_softmax = torch.softmax(patch_pred_logits, dim=1)  ## n x cls

            _, sort_idx = torch.sort(patch_pred_softmax[:, -1], descending=True)
            topk_idx_max = sort_idx[:instance_per_group].long()
            topk_idx_min = sort_idx[-instance_per_group:].long()
            topk_idx = torch.cat([topk_idx_max, topk_idx_min], dim=0)

            MaxMin_inst_feat = mid_feat.index_select(dim=0, index=topk_idx)  ##########################
            max_inst_feat = mid_feat.index_select(dim=0, index=topk_idx_max)
            af_inst_feat = att_feat_tensor

            if distill == 'MaxMinS':
                slide_pseudo_feat.append(MaxMin_inst_feat)
            elif distill == 'MaxS':
                slide_pseudo_feat.append(max_inst_feat)
            elif distill == 'AFS':
                slide_pseudo_feat.append(af_inst_feat)
            else:
                raise AssertionError(f'Error! Bad param distill = {distill}')
            #

        slide_pseudo_feat = torch.cat(slide_pseudo_feat, dim=0)  ### numGroup x fs

        ## optimization for the first tier
        slide_sub_preds = torch.cat(slide_sub_preds, dim=0)  ### numGroup x fs
        # slide_sub_labels = torch.cat(slide_sub_labels, dim=0)  ### numGroup

        # loss0 = F.cross_entropy(slide_sub_preds, slide_sub_labels).mean()

        # optimizer0.zero_grad()
        # loss0.backward(retain_graph=True)
        # torch.nn.utils.clip_grad_norm_(dimReduction.parameters(), params.grad_clipping)
        # torch.nn.utils.clip_grad_norm_(attention.parameters(), params.grad_clipping)
        # torch.nn.utils.clip_grad_norm_(classifier.parameters(), params.grad_clipping)
        # optimizer0.step()

        ## optimization for the second tier
        gSlidePred = self.attCls(slide_pseudo_feat.detach())
        # loss1 = F.cross_entropy(gSlidePred, label).mean()

        # optimizer1.zero_grad()
        # loss1.backward()
        # torch.nn.utils.clip_grad_norm_(UClassifier.parameters(), params.grad_clipping)
        # optimizer1.step()

        # Train_Loss0.update(loss0.item(), numGroup)
        # Train_Loss1.update(loss1.item(), 1)

        return slide_sub_preds, gSlidePred

    def forward(self, x):
        sub_preds = []
        preds = []

        for sx in x:
            sx = sx.cuda()
            sub_p, p = self.single_forward(sx)
            sub_preds.append(sub_p)
            preds.append(p)

        preds = torch.cat(preds, 0)

        return sub_preds, preds


if __name__ == '__main__':
    net = DTFD_MIL(128)
    net.eval()
    net.cpu()
    feat = torch.rand([2, 200, 128])

    o1, o2 = net(feat)

    print(o2)

