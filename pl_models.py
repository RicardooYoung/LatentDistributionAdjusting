import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import torch
import torch.nn as nn
import timm
from models.LDA import LDAModel
from models.loss import *


class LDA_PL(pl.LightningModule):
    def __init__(self, epoch, hyp):
        super().__init__()
        self.save_hyperparameters()
        self.hyp = hyp
        self.backbone = LDAModel(n_prototypes=self.hyp['n_prototypes'], n_features=self.hyp['n_features'])
        self.epoch = epoch
        self.cls_loss = nn.CrossEntropyLoss()
        self.inter_loss = InterLoss(delta=self.hyp['inter_delta'])
        self.intra_loss = IntraLoss(delta=self.hyp['intra_delta'])
        self.data_loss = DataLoss(scale=self.hyp['scale'], margin=self.hyp['margin'])

    def forward(self, x):
        # use forward for inference/predictions
        y, dist = self.backbone(x)
        return y, dist

    def cal_loss(self, y_hat, dist, y):
        loss = self.cls_loss(y_hat, y)
        pos, neg = self.backbone.read_prototype()
        loss += self.inter_loss(pos, neg) * self.hyp['inter_weight']
        loss += self.intra_loss(pos, neg) * self.hyp['intra_weight']
        loss += self.data_loss(dist, y) * self.hyp['data_weight']
        return loss

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat, dist = self.backbone(x)
        loss = self.cal_loss(y_hat, dist, y)
        self.log('train_loss', loss, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat, _ = self.backbone(x)
        loss = self.cls_loss(y_hat, y)
        self.log('val_loss', loss, on_step=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat, dist = self.backbone(x)
        loss = self.cal_loss(y_hat, dist, y)
        self.log('test_loss', loss, sync_dist=True)

    def configure_optimizers(self):
        opt = torch.optim.SGD(self.parameters(), lr=self.hyp['lr'], momentum=self.hyp['momentum'],
                              weight_decay=self.hyp['weight_decay'])
        epoch_steps = [int(self.epoch * 0.1), int(self.epoch * 0.2), int(self.epoch * 0.3), int(self.epoch * 0.4)]
        print('epoch_steps:', epoch_steps)

        def lr_step_func(epoch):
            return 0.1 ** len([m for m in epoch_steps if m <= epoch])

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer=opt, lr_lambda=lr_step_func)
        lr_scheduler = {
            'scheduler': scheduler,
            'name': 'learning_rate',
            'interval': 'epoch',
            'frequency': 1}
        return [opt], [lr_scheduler]
