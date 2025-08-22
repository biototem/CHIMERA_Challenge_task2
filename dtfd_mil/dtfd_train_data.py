from force_relative_import import enable_force_relative_import
with enable_force_relative_import():
    from ..base_train_data import *
    from .model_dtfd import *


class TrainData(TrainData):
    def __init__(self, task, net, lr=2e-4, n_cls=6, score_show_file=None, report_out_file=None, csv_pred_out_file=None, is_test=False):
        super().__init__(task=task, lr=lr, n_cls=n_cls,
                         score_show_file=score_show_file, report_out_file=report_out_file, csv_pred_out_file=csv_pred_out_file,
                         is_test=is_test)
        self.backbone = net

    def forward(self, x):
        return self.backbone(x)

    def training_step(self, batch, batch_idx):
        x, y, c, infos = batch

        batch_sub_preds, batch_preds = self.backbone(x)

        loss0 = 0.
        for idx in range(len(x)):
            L = batch_sub_preds[idx].shape[0]
            loss0 += task_loss_func(self.task, self.loss_func, self.n_cls, batch_sub_preds[idx], [y[idx]]*L)

        loss1 = task_loss_func(self.task, self.loss_func, self.n_cls, batch_preds, y)
        loss = loss0 + loss1

        self.log('train_sub_loss', value=loss0, prog_bar=True)
        self.log('train_second_loss', value=loss1, prog_bar=True)
        self.log('train_loss', value=loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        x, y, c, infos = batch

        batch_sub_preds, batch_preds = self.backbone(x)

        loss0 = 0.
        for idx in range(len(x)):
            L = batch_sub_preds[idx].shape[0]
            loss0 += task_loss_func(self.task, self.loss_func, self.n_cls, batch_sub_preds[idx], [y[idx]]*L)

        loss1 = task_loss_func(self.task, self.loss_func, self.n_cls, batch_preds, y)
        loss = loss0 + loss1

        out = batch_preds.cpu().numpy()
        y = [each_y.cpu().numpy() for each_y in y]
        loss = loss.item()

        cur_metric_cache = self._metric_cache.setdefault(dataloader_idx, [])
        cur_metric_cache.append([out, y, loss, infos])
