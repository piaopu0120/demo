import torch
import os
def save_checkpoint(args, epoch, model, auc, optimizer, lr_scheduler, logger):
    save_state = {'model': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'lr_scheduler': lr_scheduler.state_dict(),
                  'auc': auc,
                  'epoch': epoch,
                  'args': args}

    save_path = os.path.join(args.save_dir, f'ckpt_{args.model.name}_{args.backbone.name}_epoch_{epoch}_{auc:.5f}.pth')
    logger.info(f"{save_path} saving......")
    torch.save(save_state, save_path)
    logger.info(f"{save_path} saved !!!")

def load_checkpoint(args, model, optimizer, lr_scheduler, logger):
    logger.info(f"==============> Resuming form {args.resume}....................")
    checkpoint = torch.load(args.resume, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    logger.info(msg)
    auc = 0.0
    if 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        logger.info(f"=> loaded successfully '{args.resume}' (epoch {checkpoint['epoch']})")
        if 'auc' in checkpoint:
            auc = checkpoint['auc']
    del checkpoint
    torch.cuda.empty_cache()
    return auc