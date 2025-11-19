from architecture.transformer import AttnMIL6 as AttnMIL
from torch import nn
from t1_feat_dataset import FeatDataset_train,FeatDataset_test
from tqdm import tqdm
from sklearn.metrics import f1_score, auc, roc_curve
import pandas as pd
import random
import numpy as np
import torch
import t1_cfg
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
def seed_everything(seed_value: int):
    r"""Sets the seed for generating random numbers in :pytorch:`PyTorch`,
    :obj:`numpy` and Python.

    Args:
        seed (int): The desired seed.
    """
    np.random.seed(seed_value)
    random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)  # 为了禁止hash随机化，使得实验可复现。
    torch.manual_seed(seed_value)  # 为CPU设置随机种子
    torch.cuda.manual_seed(seed_value)  # 为当前GPU设置随机种子（只用一块GPU）
    torch.cuda.manual_seed_all(seed_value)  # 为所有GPU设置随机种子（多块GPU）
    torch.backends.cudnn.deterministic = True
def Find_Optimal_Cutoff(TPR, FPR, threshold):
  y = TPR - FPR
  Youden_index = np.argmax(y)
  optimal_threshold = threshold[Youden_index]
  point = [FPR[Youden_index], TPR[Youden_index]]
  return optimal_threshold, point

seed111 = 3407
seed_everything(seed111)




train_ds = FeatDataset_train(t1_cfg.train_data_dict)
valid_ds = FeatDataset_test(t1_cfg.val_data_dict)

train_dl = torch.utils.data.DataLoader(
    train_ds,
    batch_size=1,
    shuffle=True,
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
)

val_dl = torch.utils.data.DataLoader(
    valid_ds,
    batch_size=1,
    shuffle=False,
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
)


best_val_F1 = 0
best_val_AUC = 0

best_F1_0 = []
best_F1_1 = []
best_F1_2 = []
best_F1_3 = []
best_F1_4 = []

best_auc_0 = []
best_auc_1 = []
best_auc_2 = []
best_auc_3 = []
best_auc_4 = []

n_token = 5
model = AttnMIL(D_feat=t1_cfg.feat_dim, n_class=t1_cfg.n_cls, n_token=n_token)
model.cuda()
criterion = nn.CrossEntropyLoss()
optimizer0 = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=t1_cfg.lr, weight_decay=t1_cfg.weight_decay)
for epoch in tqdm(range(t1_cfg.epoch)):
    model.train()
    for feat_tmp, coord_tmp, labels_tmp ,linc_feat in train_dl:
        feat_tmp, coord_tmp ,labels_tmp = feat_tmp.cuda(), coord_tmp.cuda(),labels_tmp.cuda()
        linc_feat = linc_feat.cuda()
        sub_preds, slide_preds, attn = model(feat_tmp,linc_feat, use_attention_mask=True)  ###注意这里的use_attention_mask是True
        loss0 = criterion(sub_preds, labels_tmp.repeat_interleave(n_token))
        loss1 = criterion(slide_preds, labels_tmp)
        diff_loss = torch.tensor(0,dtype=torch.float32).cuda()
        attn = torch.softmax(attn, dim=-1)
        for i in range(n_token):
            for j in range(i + 1, n_token):
                diff_loss += torch.cosine_similarity(attn[:, i], attn[:, j], dim=-1).mean() / (n_token * (n_token - 1) / 2)
        loss = diff_loss + loss0 + loss1
        optimizer0.zero_grad()
        loss.backward()
        optimizer0.step()

    list_eval_label = []
    list_eval_pred = []
    #################### 评估
    model.eval()
    with torch.no_grad():
        for feat_tmp1, coord_tmp1, labels_tmp1 ,linc_feat  in val_dl:
            feat_tmp1, coord_tmp1, labels_tmp1 = feat_tmp1.cuda(), coord_tmp1.cuda(), labels_tmp1.cuda()
            linc_feat = linc_feat.cuda()
            sub_preds, slide_preds, attn = model(feat_tmp1,linc_feat, use_attention_mask=False)  ###注意这里的use_attention_mask是False
            pred = torch.softmax(slide_preds, dim=-1).cpu()
            类别1的概率 = float(pred[0][1])
            list_eval_label.append(int(labels_tmp1))
            list_eval_pred.append(类别1的概率)
    fpr, tpr, thresholds = roc_curve(list_eval_label, list_eval_pred, pos_label=None)
    val_auc_best_thresholds, optimal_point = Find_Optimal_Cutoff(TPR=tpr, FPR=fpr, threshold=thresholds)

    eval_auc = auc(fpr, tpr)
    eval_auc = round(eval_auc,4)
    pred_label_list = [1 if p > 0.5 else 0 for p in list_eval_pred]
    eval_mean_f1 = round(f1_score(list_eval_label, pred_label_list),4)
    eval_1_f1 = round(f1_score(list_eval_label, pred_label_list,average='macro'),4)
    if eval_mean_f1 > (best_val_F1 - 0.03):
        if eval_mean_f1 > (best_val_F1):
            best_val_F1 = eval_mean_f1

        print(epoch, eval_auc, eval_mean_f1, eval_1_f1)
        best_val_F1 = eval_mean_f1
        path111 = t1_cfg.ck_dir + '/seed_' + str(seed111) + '/best_1_F1/' + str(epoch) + '_' + str(
            round(best_val_F1, 2)) + '_' + str(round(val_auc_best_thresholds, 2)) + '.pth'
        path222 = t1_cfg.ck_dir + '/seed_' + str(seed111) + '/best_1_F1/' + 'score.csv'

        os.makedirs(os.path.dirname(path111), exist_ok=True)
        torch.save(model.state_dict(), path111)
        best_F1_0.append(epoch)
        best_F1_1.append(best_val_F1)
        best_F1_2.append(eval_1_f1)
        best_F1_3.append(eval_auc)
        best_F1_4.append(val_auc_best_thresholds)
        df_test_new11 = pd.DataFrame({
            'epoch_name': pd.Series(best_F1_0),
            'mean_F1': pd.Series(best_F1_1),
            '1_F1': pd.Series(best_F1_2),
            'AUC': pd.Series(best_F1_3),
            'AUC_best_thresholds': pd.Series(best_F1_4),

        })
        df_test_new11.to_csv(path222, encoding='GB18030', index=False)

