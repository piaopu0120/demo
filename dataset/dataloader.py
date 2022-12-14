import torch
import numpy as np
from .image_dataset import DeepFakeImageDataset
from .ff_image_dataset import FaceForensicsDataset
from .transforms.transform import create_transform_resize
from .transforms.gffd_transform import create_train_transforms as create_train_transform_gffd
from .transforms.gffd_transform import create_val_transforms as create_val_transform_gffd


def get_transform(conf, mode='train'):
    if mode == 'train':
        if conf.transform == 'resize':
            return create_transform_resize(conf.image_size)
        elif conf.transform == 'gffd':
            return create_train_transform_gffd(conf.image_size)
    if mode == 'val':
        if conf.transform == 'gffd':
            return create_val_transform_gffd(conf.image_size)
        return create_transform_resize(conf.image_size)


def create_train_image_dataset(conf, args):
    dataset_train_pos = DeepFakeImageDataset(
        data_file=conf.train_pos_data_path,
        mode='train',
        transform=get_transform(conf, 'train'),
        img_size=conf.img_size,
    )
    train_pos_sampler = None
    if args.distributed:
        train_pos_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset_train_pos)

    dataset_train_neg = DeepFakeImageDataset(
        data_file=conf.train_neg_data_path,
        mode='train',
        transform=get_transform(conf, 'train'),
        img_size=conf.img_size,
    )
    train_neg_sampler = None
    if args.distributed:
        train_neg_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset_train_neg)

    dataset_val = DeepFakeImageDataset(
        data_file=conf.val_data_path,
        mode='train',
        transform=get_transform(conf, 'val'),
        img_size=conf.img_size,
    )
    val_sampler = None
    if args.distributed:
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset_val)
    return dataset_train_pos, train_pos_sampler, dataset_train_neg, train_neg_sampler, dataset_val, val_sampler


def create_test_image_dataset(conf, args):
    dataset_val = DeepFakeImageDataset(
        data_file=conf.val_data_path,
        mode='train',
        transform=get_transform(conf, 'val'),
        img_size=conf.img_size,
    )
    val_sampler = None
    if args.distributed:
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset_val)
    return dataset_val, val_sampler


def create_train_FF_image_dataset(conf, args):
    dataset_train_pos = FaceForensicsDataset(is_real=True, frame_dir=conf.frame_dir, num_frames=conf.num_frames,
                                             split_dir=conf.split_dir, split='train', bbox_name=conf.bbox_name, transform=get_transform(conf, 'train'),size=conf.image_size)
    train_pos_sampler = None
    if args.distributed:
        train_pos_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset_train_pos)

    dataset_train_neg = FaceForensicsDataset(is_real=False, frame_dir=conf.frame_dir, num_frames=conf.num_frames,
                                             split_dir=conf.split_dir, split='train',  bbox_name=conf.bbox_name, transform=get_transform(conf, 'train'),size=conf.image_size)
    train_neg_sampler = None
    if args.distributed:
        train_neg_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset_train_neg)

    dataset_val = FaceForensicsDataset(is_real=True, frame_dir=conf.frame_dir, num_frames=conf.num_frames,
                                       split_dir=conf.split_dir, split='val',  bbox_name=conf.bbox_name, transform=get_transform(conf, 'val'),size=conf.image_size)
    val_sampler = None
    if args.distributed:
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset_val)
    return dataset_train_pos, train_pos_sampler, dataset_train_neg, train_neg_sampler, dataset_val, val_sampler


def create_test_FF_image_dataset(conf, args):
    dataset_val = FaceForensicsDataset(is_real=True, frame_dir=conf.frame_dir, num_frames=conf.num_frames,
                                       split_dir=conf.split_dir, split='test',  bbox_name=conf.bbox_name, transform=get_transform(conf, 'val'),size=conf.image_size)
    val_sampler = None
    if args.distributed:
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset_val)
    return dataset_val, val_sampler


def create_train_dataloader(args, dataset_train_pos, train_pos_sampler, dataset_train_neg, train_neg_sampler, dataset_val, val_sampler):
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


def create_test_dataloader(args, dataset_val, val_sampler):

    val_dataloader = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size * 2,
        num_workers=args.num_workers//2,
        shuffle=val_sampler is None,
        pin_memory=False,
        collate_fn=dataset_val.collate_val_function,
        sampler=val_sampler)
    return val_dataloader
