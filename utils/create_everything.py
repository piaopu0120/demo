# --------------------------------------------------------
# (create_logger)Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

from webbrowser import get
from torch import optim
from models.base_net import BaseNet
from dataset import dataloader
import os
import sys
import logging
import functools
from termcolor import colored
from models.resnet_model import ResNet
from models.resnet_srm import ResNet_srm
from models.resnet_dct import ResNet_dct
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


def create_dataloader(args, logger):
    if args.mode == 'train' and args.object == 'image':
        if args.dataset.name in ['FF_face', 'small']:
            dataset_train_pos, train_pos_sampler, dataset_train_neg, train_neg_sampler, dataset_val, val_sampler = dataloader.create_train_image_dataset(
                getattr(args.dataset, args.dataset.name), args)
        elif args.dataset.name in ['FF_dir']:
            dataset_train_pos, train_pos_sampler, dataset_train_neg, train_neg_sampler, dataset_val, val_sampler = dataloader.create_train_FF_image_dataset(
                getattr(args.dataset, args.dataset.name), args)
        logger.info(f'loading train pos:{len(dataset_train_pos)}')
        logger.info(f'loading train neg:{len(dataset_train_neg)}')
        logger.info(f'loading val :{len(dataset_val)}')
        return dataloader.create_train_dataloader(args, dataset_train_pos, train_pos_sampler, dataset_train_neg, train_neg_sampler, dataset_val, val_sampler)
    if args.mode == 'test' and args.object == 'image':
        if args.dataset.name in ['FF_face', 'small']:
            dataset_val, val_sampler = dataloader.create_test_image_dataset(
                getattr(args.dataset, args.dataset.name), args)
        elif args.dataset.name in ['FF_dir']:
            dataset_val, val_sampler = dataloader.create_test_FF_image_dataset(
                getattr(args.dataset, args.dataset.name), args)
        logger.info(f'loading val :{len(dataset_val)}')
        return dataloader.create_test_dataloader(args, dataset_val, val_sampler)


def create_model(args):
    if args.model.name == 'base_net':
        return BaseNet(args.backbone.name, args.pretrained, getattr(args.backbone, args.backbone.name))
    elif args.model.name == 'resnet':
        return ResNet()
    elif args.model.name == 'srm':
        return ResNet_srm()
    elif args.model.name == 'dct':
        return ResNet_dct(args.pretrained,args.image_size)


def create_optimizer(opt_name, model, conf):
    optimizer = None
    if opt_name == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),  lr=conf.lr, weight_decay=conf.weight_decay)
    elif opt_name == 'adam':
        optimizer = optim.Adam(
            model.parameters(), lr=conf.lr, betas=(conf.betas1, conf.betas2), eps=conf.eps, weight_decay=conf.weight_decay
        )
    elif opt_name == 'SGD':
        optimizer = optim.SGD(
            model.parameters(), lr=conf.lr, momentum=conf.momentum, weight_decay=conf.decay
        )
    return optimizer


def create_scheduler(sche_name, optimizer, conf):
    scheduler = None
    if sche_name == 'MultiStepLR':
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=conf.milestones, gamma=conf.gamma)
    elif sche_name == 'StepLR':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=conf.step_size, gamma=conf.gamma, last_epoch=-1, verbose=False)
    return scheduler
