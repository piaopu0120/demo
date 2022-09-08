# --------------------------------------------------------
# (create_logger)Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

from webbrowser import get
from torch import optim
from models.base_net import BaseNet
from models.base_net_gj import BinaryClassifier
from dataset import dataloader
import os
import sys
import logging
import functools
from termcolor import colored
sys.path.append('..')


@functools.lru_cache()
def create_logger(output_dir, dist_rank=0, name=''):
    # create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # create formatter
    fmt = '[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s'
    color_fmt = colored('[%(asctime)s %(name)s]', 'green') + \
        colored('(%(filename)s %(lineno)d)', 'yellow') + \
        ': %(levelname)s %(message)s'

    # create console handlers for master process
    if dist_rank == 0:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(
            logging.Formatter(fmt=color_fmt, datefmt='%Y-%m-%d %H:%M:%S'))
        logger.addHandler(console_handler)

    # create file handlers
    file_handler = logging.FileHandler(os.path.join(
        output_dir, f'log_rank{dist_rank}.txt'), mode='a')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(file_handler)

    return logger


def create_dataloader(args):
    if args.mode == 'train' and args.object == 'image':
        # train_pos,train_neg,val
        return dataloader.create_train_image_dataloader(getattr(args.dataset, args.dataset.name), args)


def create_model(args):
    if args.model.name == 'base_net':
        return BaseNet(args.backbone.name, getattr(args.backbone, args.backbone.name))


def create_optimizer(opt_name, model, args):
    optimizer = None
    if opt_name == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),  lr=args.lr, weight_decay=args.weight_decay)
    return optimizer


def create_scheduler(sche_name, optimizer, args):
    scheduler = None
    if sche_name == 'MultiStepLR':
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=args.milestones, gamma=args.gamma)
    return scheduler
