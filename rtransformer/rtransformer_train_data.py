import time

from force_relative_import import enable_force_relative_import
with enable_force_relative_import():
    from ..base_train_data import *


class TrainData(TrainData):
    def __init__(self, task, net, lr=2e-4, n_cls=6, score_show_file=None, report_out_file=None, csv_pred_out_file=None, is_test=False):
        super().__init__(task=task, lr=lr, n_cls=n_cls,
                         score_show_file=score_show_file, report_out_file=report_out_file, csv_pred_out_file=csv_pred_out_file,
                         is_test=is_test)
        self.backbone = net

    def forward(self, x):
        # 判断张量中是否存在NaN
        has_nan = torch.isnan(x[0]).any().item()

        # 输出结果
        if has_nan:
            print("张量中存在NaN")
            print(x)
            time.sleep(1000)


        return self.backbone(x)

    def forward_for_extract_heat(self, x):
        o = self.forward(x)
        return o

    def training_step(self, batch, batch_idx):
        x, y, c, infos = batch
        out = self.forward(x)

        loss = task_loss_func(self.task, self.loss_func, self.n_cls, out, y)

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        x, y, c, infos = batch

        out = self.forward(x)

        loss = task_loss_func(self.task, self.loss_func, self.n_cls, out, y)
        loss = loss.item()

        out = out.cpu().numpy()
        y = [each_y.cpu().numpy() for each_y in y]

        cur_metric_cache = self._metric_cache.setdefault(dataloader_idx, [])
        cur_metric_cache.append([out, y, loss, infos])
