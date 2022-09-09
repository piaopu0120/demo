import os
import torch
import torch.distributed as dist
import time
import datetime
import json
from utils.create_everything import create_model, create_dataloader, create_logger, create_optimizer, create_scheduler
from configs import load_configs
from engine import train_one_epoch, validate
import numpy as np
import random
import torch.backends.cudnn as cudnn
from torch.nn import DataParallel
from utils.tools import save_checkpoint,load_checkpoint

def main(args):
    logger.info(f"Creating model {args.model.name}, backbone: {args.backbone.name}")
    model = create_model(args)
    model.cuda() 
    optimizer = create_optimizer(args.opt.name, model,getattr(args.opt,args.opt.name))

    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(
            model)  # 跨卡同步BN，训练效果不受使用GPU数量影响，提高模型收敛速度
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[
                                                        args.local_rank], broadcast_buffers=False, find_unused_parameters=True)
    else:
        model = DataParallel(model)
    val_dataloader = create_dataloader(args)

    lr_scheduler = create_scheduler(args.sched.name,optimizer,getattr(args.sched,args.sched.name))

    criterion = None 
    if args.criterion == 'CE':
        criterion = torch.nn.CrossEntropyLoss()  # torch.nn.BCEWithLogitsLoss()

    max_auc = 0.0

    if args.load_checkpoint and args.resume:
        max_auc = load_checkpoint(args, model, optimizer, lr_scheduler, logger)

    logger.info("Start inferencing...")
    start_time = time.time()
  
    auc = validate(args, val_dataloader, model,
                    criterion, logger, 0, max_auc)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Inference time {}'.format(total_time_str))
        

if __name__ == '__main__':
    args = load_configs.get_parameters()

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        args.local_rank = int(os.environ["LOCAL_RANK"])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
        args.distributed = True
    else:
        rank = -1
        world_size = -1
        args.distributed = False
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(
        backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier()

    seed = args.manual_seed + dist.get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    os.makedirs(args.save_dir, exist_ok=True)
    logger = create_logger(output_dir=args.save_dir,
                           dist_rank=dist.get_rank(), name=f"{args.model.name}")

    # print config
    logger.info(vars(args))

    main(args)
