import torch
import numpy as np
from .image_dataset import DeepFakeImageDataset
from .transform import create_transform_resize


def get_transform(conf, mode='train'):
    if mode == 'train':
        if conf.transform == 'resize':
            return create_transform_resize(conf.image_size)
    if mode == 'val':
        return create_transform_resize(conf.image_size)


def create_train_image_dataloader(conf, args):
    dataset_train_pos = DeepFakeImageDataset(
        data_file=conf.train_pos_data_path,
        mode='train',
        transform=get_transform(conf, 'train'),
        conf=conf,
    )
    train_pos_sampler = None
    if args.distributed:
        train_pos_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset_train_pos)

    dataset_train_neg = DeepFakeImageDataset(
        data_file=conf.train_neg_data_path,
        mode='train',
        transform=get_transform(conf, 'train'),
        conf=conf,
    )
    train_neg_sampler = None
    if args.distributed:
        train_neg_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset_train_neg)

    dataset_val = DeepFakeImageDataset(
        data_file=conf.val_data_path,
        mode='train',
        transform=get_transform(conf, 'val'),
        conf=conf,
    )
    val_sampler = None
    if args.distributed:
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset_val)

    train_pos_dataloader = torch.utils.data.DataLoader(
        dataset_train_pos,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=train_pos_sampler is None,
        pin_memory=False,
        collate_fn=dataset_train_pos.collate_train_function,
        sampler=train_pos_sampler)

    train_neg_dataloader = torch.utils.data.DataLoader(
        dataset_train_neg,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=train_neg_sampler is None,
        pin_memory=False,
        collate_fn=dataset_train_neg.collate_train_function,
        sampler=train_neg_sampler)

    val_dataloader = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size * 2,
        num_workers=args.num_workers//2,
        shuffle=val_sampler is None,
        pin_memory=False,
        collate_fn=dataset_val.collate_val_function,
        sampler=val_sampler)
    return [train_pos_dataloader, train_neg_dataloader, val_dataloader], [train_pos_sampler, train_neg_sampler]
