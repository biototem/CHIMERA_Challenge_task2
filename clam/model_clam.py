import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            m.bias.data.zero_()

        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


def create_positive_targets(length, device):
    return torch.full([length], 1, device=device, dtype=torch.int64)


def create_negative_targets(length, device):
    return torch.full([length], 0, device=device, dtype=torch.int64)


"""
Attention Network without Gating (2 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""


class Attn_Net(nn.Module):

    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        super(Attn_Net, self).__init__()
        self.module = [
            nn.Linear(L, D),
            nn.Tanh()]

        if dropout:
            self.module.append(nn.Dropout(0.25))

        self.module.append(nn.Linear(D, n_classes))

        self.module = nn.Sequential(*self.module)

    def forward(self, x):
        return self.module(x), x  # N x n_classes


class Attn_Net_Gated(nn.Module):
    """
    Attention Network with Sigmoid Gating (3 fc layers)
    args:
        L: input feature dimension
        D: hidden layer dimension
        dropout: whether to use dropout (p = 0.25)
        n_classes: number of classes
    """

    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]

        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)

        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x


class CLAM_SB(nn.Module):
    """
    args:
        gate: whether to use gated attention network
        size_arg: config for network size
        dropout: whether to use dropout
        k_sample: number of positive/neg patches to sample for instance-level training
        dropout: whether to use dropout (p = 0.25)
        n_classes: number of classes
        instance_loss_fn: loss function to supervise instance-level training
        subtyping: whether it's a subtyping problem
    """

    def __init__(self, feat_dim=1024, gate=True, size_arg="small", dropout=False, k_sample=8, n_classes=2,
                 instance_loss_fn=nn.CrossEntropyLoss(), subtyping=False):
        super(CLAM_SB, self).__init__()
        self.size_dict = {"small": [feat_dim, 512, 256],
                          "big": [feat_dim, 512, 384]}
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]

        if dropout:
            fc.append(nn.Dropout(0.25))

        if gate:
            attention_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=dropout, n_classes=1)
        else:
            attention_net = Attn_Net(L=size[1], D=size[2], dropout=dropout, n_classes=1)

        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.classifiers = nn.Linear(size[1], n_classes)
        instance_classifiers = [nn.Linear(size[1], 2) for i in range(n_classes)]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)
        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        self.n_classes = n_classes
        self.subtyping = subtyping

        initialize_weights(self)

    # instance-level evaluation for in-the-class attention branch
    def inst_eval(self, A, h, classifier):
        device = h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        k_sample = min(self.k_sample, A.shape[-1])
        top_p_ids = torch.topk(A, k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        top_n_ids = torch.topk(-A, k_sample, dim=1)[1][-1]
        top_n = torch.index_select(h, dim=0, index=top_n_ids)
        p_targets = create_positive_targets(k_sample, device)
        n_targets = create_negative_targets(k_sample, device)

        all_targets = torch.cat([p_targets, n_targets], dim=0)
        all_instances = torch.cat([top_p, top_n], dim=0)
        logits = classifier(all_instances)
        all_preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)
        if self.instance_loss_fn is None:
            instance_loss = -1
        else:
            instance_loss = self.instance_loss_fn(logits, all_targets)
        return instance_loss, all_preds, all_targets

    # instance-level evaluation for out-of-the-class attention branch
    def inst_eval_out(self, A, h, classifier):
        device = h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        p_targets = create_negative_targets(self.k_sample, device)
        logits = classifier(top_p)
        p_preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)
        if self.instance_loss_fn is None:
            instance_loss = -1
        else:
            instance_loss = self.instance_loss_fn(logits, p_targets)
        return instance_loss, p_preds, p_targets

    def single_forward(self, x, label=None, instance_eval=False):
        if x.shape[0] < self.k_sample:
            x_pad = torch.randn([self.k_sample-x.shape[0], x.shape[1]], dtype=x.dtype, device=x.device)
            x = torch.cat([x, x_pad], 0)

        A, x = self.attention_net(x)  # NxK
        A_raw = torch.transpose(A, 1, 0)  # KxN

        A = F.softmax(A_raw, dim=1)  # softmax over N

        if instance_eval:
            total_inst_loss = 0.0
            all_preds = []
            all_targets = []
            inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze()  # binarize label

            for i in range(len(self.instance_classifiers)):
                inst_label = inst_labels[i].item()
                classifier = self.instance_classifiers[i]
                if inst_label == 1:  # in-the-class:
                    instance_loss, preds, targets = self.inst_eval(A, x, classifier)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())

                else:  # out-of-the-class
                    if self.subtyping:
                        instance_loss, preds, targets = self.inst_eval_out(A, x, classifier)
                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(targets.cpu().numpy())
                    else:
                        continue

                total_inst_loss += instance_loss

            if self.subtyping:
                total_inst_loss /= len(self.instance_classifiers)

            inst_loss = total_inst_loss
            inst_labels = np.asarray(all_targets)
            inst_preds = np.asarray(all_preds)

        else:
            inst_loss = None
            inst_labels = None
            inst_preds = None

        M = torch.mm(A, x)
        logit = self.classifiers(M)
        hat = torch.topk(logit, 1, dim=1)[1]
        prob = F.softmax(logit, dim=1)

        inter_feat = M

        return logit, prob, hat, A_raw, inter_feat, inst_loss, inst_labels, inst_preds

    def forward(self, x, y=None, instance_eval=False):
        if y is None:
            y = [None] * len(x)

        batch_logit = []
        batch_prob = []
        batch_hat = []
        batch_A_raw = []
        batch_inter_feat = []
        batch_inst_loss = []
        batch_inst_labels = []
        batch_inst_preds = []

        for xi, yi in zip(x, y):
            logits, Y_prob, Y_hat, A_raw, inter_feat, inst_loss, inst_labels, inst_preds = self.single_forward(xi, yi, instance_eval=instance_eval)
            batch_logit.append(logits)
            batch_prob.append(Y_prob)
            batch_hat.append(Y_hat)
            batch_A_raw.append(A_raw)
            batch_inter_feat.append(inter_feat)
            batch_inst_loss.append(inst_loss)
            batch_inst_labels.append(inst_labels)
            batch_inst_preds.append(inst_preds)

        batch_logit = torch.cat(batch_logit, 0)
        batch_prob = torch.cat(batch_prob, 0)
        batch_hat = torch.cat(batch_hat, 0)
        if batch_inst_loss[0] is not None:
            batch_inst_loss = torch.stack(batch_inst_loss, 0)

        return batch_logit, batch_prob, batch_hat, batch_A_raw, batch_inter_feat, batch_inst_loss, batch_inst_labels, batch_inst_preds


class CLAM_MB(CLAM_SB):
    def __init__(self, feat_dim=1024, gate=True, size_arg="small", dropout=False, k_sample=8, n_classes=2,
                 instance_loss_fn=nn.CrossEntropyLoss(), subtyping=False):
        nn.Module.__init__(self)
        self.size_dict = {"small": [feat_dim, 512, 256],
                          "big": [feat_dim, 512, 384]}
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))
        if gate:
            attention_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=dropout, n_classes=n_classes)
        else:
            attention_net = Attn_Net(L=size[1], D=size[2], dropout=dropout, n_classes=n_classes)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        bag_classifiers = [nn.Linear(size[1], 1) for i in range(n_classes)]  # use an indepdent linear layer to predict each class
        self.classifiers = nn.ModuleList(bag_classifiers)
        instance_classifiers = [nn.Linear(size[1], 2) for i in range(n_classes)]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)
        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        self.n_classes = n_classes
        self.subtyping = subtyping
        initialize_weights(self)

    def single_forward(self, x, label=None, instance_eval=False):
        if x.shape[0] < self.k_sample:
            x_pad = torch.randn([self.k_sample-x.shape[0], x.shape[1]], dtype=x.dtype, device=x.device)
            x = torch.cat([x, x_pad], 0)

        A, x = self.attention_net(x)  # NxK
        A_raw = torch.transpose(A, 1, 0)  # KxN

        A = F.softmax(A_raw, dim=1)  # softmax over N

        if instance_eval:
            total_inst_loss = 0.0
            all_preds = []
            all_targets = []
            inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze()  # binarize label
            for i in range(len(self.instance_classifiers)):
                inst_label = inst_labels[i].item()
                classifier = self.instance_classifiers[i]
                if inst_label == 1:  # in-the-class:
                    instance_loss, preds, targets = self.inst_eval(A[i], x, classifier)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                else:  # out-of-the-class
                    if self.subtyping:
                        instance_loss, preds, targets = self.inst_eval_out(A[i], x, classifier)
                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(targets.cpu().numpy())
                    else:
                        continue
                total_inst_loss += instance_loss

            if self.subtyping:
                total_inst_loss /= len(self.instance_classifiers)

            inst_loss = total_inst_loss
            inst_labels = np.asarray(all_targets)
            inst_preds = np.asarray(all_preds)

        else:
            inst_loss = None
            inst_labels = None
            inst_preds = None

        M = torch.mm(A, x)
        logit = torch.empty(1, self.n_classes, dtype=torch.float32, device=x.device)

        for c in range(self.n_classes):
            logit[0, c] = self.classifiers[c](M[c])

        hat = torch.topk(logit, 1, dim=1)[1]
        prob = F.softmax(logit, dim=1)

        inter_feat = M

        return logit, prob, hat, A_raw, inter_feat, inst_loss, inst_labels, inst_preds
