from argparse import ArgumentParser

import os
import os.path as osp
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import timm
from datasets.dataset import MyDataset
from pl_models import *
import json





def main():
    pl.seed_everything(727)

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--backbone', default='resnet50d', type=str)
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--epoch', default=10, type=int)
    parser.add_argument('--root', default='data', type=str)
    parser.add_argument('--hyp', default='hyp/LDA_V1.json', type=str)
    parser.add_argument('--num-gpus', default=1, type=int)
    parser.add_argument('--tf32', action='store_true')
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    if not args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    else:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    # ------------
    # data
    # ------------
    train_set = MyDataset(root_dir=args.root, split='train')
    val_set = MyDataset(root_dir=args.root, split='val')
    print('train data size:', len(train_set))

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True,
                              persistent_workers=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=8, shuffle=False, persistent_workers=True)

    # ------------
    # hyp
    # ------------
    f = open(args.hyp, 'r')
    content = f.read()
    hyp = json.loads(content)

    # ------------
    # model
    # ------------
    model = LDA_PL(epoch=args.epoch, hyp=hyp)
    ckpt_path = 'results'
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
    exists_results = len(os.listdir(ckpt_path))
    ckpt_path = os.path.join(ckpt_path, str(exists_results))
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

    # ------------
    # training
    # ------------
    checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',
            dirpath=ckpt_path,
            filename='{epoch:02d}-{val_loss:.6f}',
            save_top_k=10,
            mode='min',
            )
    lr_monitor = LearningRateMonitor(logging_interval='step')
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=args.num_gpus,
        benchmark=True,
		logger=TensorBoardLogger(osp.join(ckpt_path, 'logs')),
        callbacks=[checkpoint_callback, lr_monitor],
        check_val_every_n_epoch=1,
        max_epochs=args.epoch,
        strategy='ddp_find_unused_parameters_false',
    )
    trainer.fit(model, train_loader, val_loader)

if __name__ == '__main__':
    main()

