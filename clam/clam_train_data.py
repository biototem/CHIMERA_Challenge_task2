from force_relative_import import enable_force_relative_import
with enable_force_relative_import():
    from ..base_train_data import *
    from .model_clam import CLAM_MB, CLAM_SB
    from .topk import SmoothTop1SVM


class TrainData(TrainData):
    def __init__(self, task, clam_kwargs, lr=2e-4, n_cls=6, score_show_file=None, report_out_file=None, csv_pred_out_file=None, is_test=False):
        super().__init__(task=task, lr=lr, n_cls=n_cls,
                         score_show_file=score_show_file, report_out_file=report_out_file, csv_pred_out_file=csv_pred_out_file,
                         is_test=is_test)
        is_sb = clam_kwargs.pop('is_sb')
        inst_loss_func = clam_kwargs.pop('inst_loss_func')

        # bag_loss_func = 'ce'
        # if bag_loss_func == 'ce':
        #     self.loss_func = nn.CrossEntropyLoss(label_smoothing=0.2)
        # elif bag_loss_func == 'svm':
        #     self.loss_func = SmoothTop1SVM(clam_kwargs['n_classes'])
        # else:
        #     raise AssertionError('Error! Unknow bag_loss_func')

        if inst_loss_func == 'ce':
            self.inst_loss_func = nn.CrossEntropyLoss(label_smoothing=0.2)
        elif inst_loss_func == 'svm':
            self.inst_loss_func = SmoothTop1SVM(2)
        else:
            raise AssertionError('Error! Unknow inst_loss_func')

        if is_sb:
            self.backbone = CLAM_SB(**clam_kwargs, instance_loss_fn=self.inst_loss_func)
        else:
            self.backbone = CLAM_MB(**clam_kwargs, instance_loss_fn=self.inst_loss_func)

        self.bag_weight = 0.3

    def forward(self, x):
        return self.backbone(x)

    def forward_for_extract_heat(self, x):
        batch_logit, batch_prob, batch_hat, batch_A_raw, batch_inter_feat, batch_inst_loss, batch_inst_labels, batch_inst_preds =\
            self.backbone(x, None, instance_eval=False)
        return batch_logit

    def forward_for_extract_heat_2(self, x, target_cls=None):
        As = []

        batch_logit, batch_prob, batch_hat, batch_A_raw, batch_inter_feat, batch_inst_loss, batch_inst_labels, batch_inst_preds =\
            self.backbone(x, None, instance_eval=False)

        batch_hat = batch_hat.cpu().numpy()

        for A, hat in zip(batch_A_raw, batch_hat):
            if target_cls is None:
                    target_cls = hat
            if len(A) == 1:
                # clamsb
                A = A[0]
            else:
                # clammb
                A = A[target_cls]
            As.append(A)

        return batch_logit, As

    def training_step(self, batch, batch_idx):
        x, y, c, infos = batch

        if self.task in ('regression', 'regression-mse-ord-r03-z200', 'regression-ord-r03-z100'):
            instance_eval = False
        else:
            instance_eval = True

        batch_logit, batch_prob, batch_hat, batch_A_raw, batch_inter_feat, batch_inst_loss, batch_inst_labels, batch_inst_preds = self.backbone(x, y, instance_eval=instance_eval)

        bag_loss = task_loss_func(self.task, self.loss_func, self.n_cls, batch_logit, y)
        if instance_eval:
            inst_loss = torch.mean(batch_inst_loss)
            loss = self.bag_weight * bag_loss + (1 - self.bag_weight) * inst_loss
        else:
            inst_loss = 0.
            loss = bag_loss

        self.log('train_bag_loss', value=bag_loss, prog_bar=True)
        self.log('train_inst_loss', value=inst_loss, prog_bar=True)
        self.log('train_loss', value=loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        x, y, c, infos = batch

        if self.task in ('regression', 'regression-mse-ord-r03-z200', 'regression-ord-r03-z100'):
            instance_eval = False
        else:
            instance_eval = True

        batch_logit, batch_prob, batch_hat, batch_A_raw, batch_inter_feat, batch_inst_loss, batch_inst_labels, batch_inst_preds = self.backbone(x, y, instance_eval=instance_eval)

        bag_loss = task_loss_func(self.task, self.loss_func, self.n_cls, batch_logit, y)

        out = batch_logit.cpu().numpy()
        y = [each_y.cpu().numpy() for each_y in y]
        loss = bag_loss.item()

        cur_metric_cache = self._metric_cache.setdefault(dataloader_idx, [])
        cur_metric_cache.append([out, y, loss, infos])
