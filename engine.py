from utils.average_meter import AverageMeter_base
import math
import torch
import sys
import time
import datetime
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score
import numpy as np
from tqdm import tqdm
import os


def get_metric(threshold, targets, scores):
    n = len(targets)
    tn, fp, fn, tp = confusion_matrix(
        targets, scores > threshold).reshape(-1)
    assert ((tn+fp+fn+tp) == n)
    # 灵敏度(sensitivity): 正样本的recall = TPR, 一旦有fake立马被检测出来
    sen = float(tp) / (tp+fn+1e-8)
    # 特异度(specificity): 负样本的recall = TFR, real中被认为是real的概率， 对立面是误诊
    spe = float(tn) / (tn+fp+1e-8)
    f1 = 2.0 * sen * spe / (sen+spe)
    acc = float(tn+tp)/n
    scores = scores.cpu().numpy()
    scores[np.isnan(scores)] = 0

    auc = roc_auc_score(targets, scores)
    # 平均精度
    ap = average_precision_score(targets, scores)

    return sen, spe, f1, auc, ap, acc, scores


def train_one_epoch(args, model, criterion, data_pos_loader, data_neg_loader, optimizer, lr_scheduler, epoch, logger):
    model.train()
    optimizer.zero_grad()
    TrainMeter = AverageMeter_base()
    n_iter_per_epoch = min([len(dataloader)
                           for dataloader in [data_pos_loader, data_neg_loader]])
    start_time = time.time()
    for idx, datas in enumerate(zip(data_pos_loader, data_neg_loader)):
        
        labels = torch.cat([item[0] for item in datas], dim=0)
        imgs = torch.cat([item[1] for item in datas], dim=0)
        labels = labels.cuda(non_blocking=True)
        imgs = imgs.cuda(non_blocking=True)
        outputs = model(imgs)
#         if idx==0:
#             print(labels) # tensor([0, 0, 1, 1]
#             print(outputs)
#  # tensor([[-0.0196,  0.1183],
#         # [ 0.0693,  0.3164],
#         # [-0.0610,  0.2258],
#         # [ 0.0915,  0.4140]]
        loss = criterion(outputs, labels)
        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step(

        )
        torch.cuda.synchronize()

        TrainMeter.update(loss.item(), args.batch_size)
        if idx % args.print_freq == 0:
            lr = optimizer.param_groups[0]['lr']
            wd = optimizer.param_groups[0]['weight_decay']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(
                f'Train: [{epoch}/{args.epochs}][{idx}/{ n_iter_per_epoch}]\t'
                f'lr {lr:.7f}\t'
                f'wd {wd:.7f}\t'
                f'avg_loss {TrainMeter.avg_loss:.4f}\t'
                f'time_all {TrainMeter.time_all}\t'
                f'mem {memory_used:.0f}MB')
    lr_scheduler.step()

    total_time = time.time() - start_time
    logger.info(
        f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(total_time))}")


@torch.no_grad()
def validate(args, data_loader, model, criterion, logger, current_epoch, best_auc):
    model.eval()

    TestMeter = AverageMeter_base()

    targets = []
    outputs = []
    scores = []
    labels = [0, 1]
    target_names = ['real', 'fake']
    for idx, (target, images, _) in tqdm(enumerate(data_loader)):
        if idx % 100 == 0 and args.local_rank == 0:
            logger.info("Testing {}/{}".format(idx, len(data_loader)))

        images = images.cuda(non_blocking=True)
        output = model(images)

        outputs.append(output)
        score, preds = torch.softmax(output, dim=1).max(1)

        targets.append(target)
        scores.append(preds.flatten().cpu())

    targets = torch.cat(targets, dim=0).reshape(-1).long().cuda()
    scores = torch.cat(scores, dim=0).float()
    outputs = torch.cat(outputs, dim=0).float()

    loss = criterion(outputs, targets)

    TestMeter.update(loss.item(), len(targets))

    targets = targets.cpu().numpy()
    
    sen, spe, f1, auc, ap, acc, scores = get_metric(
        args.threshold, targets, scores)

    if args.local_rank == 0:
        if auc > best_auc:
            best_auc = auc
        logger.info(f"\n{classification_report(targets, scores > args.threshold,labels=labels, target_names=target_names, digits=4)}")
        logger.info(
            f"Test epoch:{current_epoch}, avg_loss:{TestMeter.avg_loss}, time_all:{TestMeter.time_all}")
        logger.info(
            f"best_auc:{best_auc}, auc:{auc}, f1:{f1}, acc:{acc}, ap:{ap}, sen:{sen}, spe:{spe}")

    return auc
