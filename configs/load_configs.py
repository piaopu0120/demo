import json
import argparse
from omegaconf import OmegaConf


def get_parameters():
    """define the parameter for training

    Args:
        --config (string): the path of config files
        --distributed (int): train the model in the mode of DDP or Not, default: 1
        --local_rank (int): define the rank of this process
        --world_size (int): define the Number of GPU
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug',action='store_true',default=False)
    parser.add_argument('-c', '--config', type=str, default='configs/base_net/train.yaml')
    parser.add_argument('--cuda',type=bool,default=True)
    # distributed setting
    parser.add_argument('--local_rank',type=int,default=0, help='local rank for DistributedDataParallel')
    parser.add_argument("--num_workers", default=16, type=int)
    # save
    parser.add_argument('--save',default=False,action='store_true')
    parser.add_argument('--save_dir',type=str,default='./save/')
    parser.add_argument('--save_name',type=str,default='loss.log')
    args = parser.parse_args()

    _C = OmegaConf.load(args.config)
    _C.merge_with(vars(args))

    if _C.debug:
        _C.train.epochs = 2

    return _C


if __name__ == '__main__':
    args = get_parameters()
    # print(args)
    print(args.dataset)
    kwargs = getattr(args.dataset,args.dataset.name)
    print(kwargs)
