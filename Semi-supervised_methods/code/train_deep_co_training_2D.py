import argparse
import logging
import os
import random
import shutil
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm

from dataloaders import utils
from dataloaders.dataset import (BaseDataSetsPro, RandomGenerator,
                                 TwoStreamBatchSampler)
from networks.discriminator import FCDiscriminator
from networks.net_factory import net_factory
from utils import losses, metrics, ramps
from val_2D import test_single_volume

#----> Metrics
import segmentation_models_pytorch as smp  ###1



# # ### consep fpn
# # CUDA_VISIBLE_DEVICES=6 python train_deep_co_training_2D.py --root_path ../data/ConSep --csv_path consep_split.csv --label_ratio 0.05 --model fpn --num_classes 5 --exp ConSep/CoTraining --pretrained /data114_2/shaozc/SegDiff/logs/scripts/consep_412/FPN_0.05_aug/epoch=194-val_IoU=0.3739.ckpt
# CUDA_VISIBLE_DEVICES=4 python Mytest.py --root_path ../data/ConSep --csv_path consep_split.csv --label_ratio 0.05 --model fpn --num_classes 5 --exp ConSep/CoTraining

# # CUDA_VISIBLE_DEVICES=6 python train_deep_co_training_2D.py --root_path ../data/ConSep --csv_path consep_split.csv --label_ratio 0.1 --model fpn --num_classes 5 --exp ConSep/CoTraining --pretrained /data114_2/shaozc/SegDiff/logs/scripts/consep_412/FPN_0.1_aug/epoch=128-val_IoU=0.5911.ckpt
# CUDA_VISIBLE_DEVICES=4 python Mytest.py --root_path ../data/ConSep --csv_path consep_split.csv --label_ratio 0.1 --model fpn --num_classes 5 --exp ConSep/CoTraining

# # CUDA_VISIBLE_DEVICES=6 python train_deep_co_training_2D.py --root_path ../data/ConSep --csv_path consep_split.csv --label_ratio 0.2 --model fpn --num_classes 5 --exp ConSep/CoTraining --pretrained /data114_2/shaozc/SegDiff/logs/scripts/consep_412/FPN_0.2_aug/epoch=133-val_IoU=0.5815.ckpt
# CUDA_VISIBLE_DEVICES=4 python Mytest.py --root_path ../data/ConSep --csv_path consep_split.csv --label_ratio 0.2 --model fpn --num_classes 5 --exp ConSep/CoTraining


# # ### consep fpn 107
# # CUDA_VISIBLE_DEVICES=6 python train_deep_co_training_2D.py --root_path ../data/ConSep --csv_path consep_split.csv --label_ratio 0.05 --model fpn --num_classes 5 --exp ConSep/CoTraining --pretrained /data114_2/shaozc/SegDiff/logs/scripts/consep_412/FPN_0.05_aug/epoch=194-val_IoU=0.3739.ckpt
# CUDA_VISIBLE_DEVICES=4 python Mytest.py --root_path ../data/ConSep --csv_path consep_split.csv --label_ratio 0.05 --model fpn --num_classes 5 --exp ConSep/CoTraining

# # CUDA_VISIBLE_DEVICES=6 python train_deep_co_training_2D.py --root_path ../data/ConSep --csv_path consep_split.csv --label_ratio 0.1 --model fpn --num_classes 5 --exp ConSep/CoTraining --pretrained /data114_2/shaozc/SegDiff/logs/scripts/consep_412/FPN_0.1_aug/epoch=128-val_IoU=0.5911.ckpt
# CUDA_VISIBLE_DEVICES=4 python Mytest.py --root_path ../data/ConSep --csv_path consep_split.csv --label_ratio 0.1 --model fpn --num_classes 5 --exp ConSep/CoTraining

# # CUDA_VISIBLE_DEVICES=6 python train_deep_co_training_2D.py --root_path ../data/ConSep --csv_path consep_split.csv --label_ratio 0.2 --model fpn --num_classes 5 --exp ConSep/CoTraining --pretrained /data114_2/shaozc/SegDiff/logs/scripts/consep_412/FPN_0.2_aug/epoch=133-val_IoU=0.5815.ckpt
# CUDA_VISIBLE_DEVICES=4 python Mytest.py --root_path ../data/ConSep --csv_path consep_split.csv --label_ratio 0.2 --model fpn --num_classes 5 --exp ConSep/CoTraining



# # ########pannuke
# # ### pannuke fpn
# # CUDA_VISIBLE_DEVICES=6 python train_deep_co_training_2D.py --root_path ../data/Pannuke --csv_path pannuke.csv --label_ratio 0.05 --model fpn --num_classes 6 --exp Pannuke/CoTraining --pretrained /data114_2/shaozc/SegDiff/logs/scripts/pannuke/FPN_0.05_aug/epoch=83-val_IoU=0.3798.ckpt
# CUDA_VISIBLE_DEVICES=4 python Mytest.py --root_path ../data/Pannuke --csv_path pannuke.csv --label_ratio 0.05 --model fpn --num_classes 6 --exp Pannuke/CoTraining

# # CUDA_VISIBLE_DEVICES=6 python train_deep_co_training_2D.py --root_path ../data/Pannuke --csv_path pannuke.csv --label_ratio 0.1 --model fpn --num_classes 6 --exp Pannuke/CoTraining --pretrained /data114_2/shaozc/SegDiff/logs/scripts/pannuke/FPN_0.1_aug/epoch=75-val_IoU=0.4960.ckpt
# CUDA_VISIBLE_DEVICES=4 python Mytest.py --root_path ../data/Pannuke --csv_path pannuke.csv --label_ratio 0.1 --model fpn --num_classes 6 --exp Pannuke/CoTraining

# # CUDA_VISIBLE_DEVICES=6 python train_deep_co_training_2D.py --root_path ../data/Pannuke --csv_path pannuke.csv --label_ratio 0.2 --model fpn --num_classes 6 --exp Pannuke/CoTraining --pretrained /data114_2/shaozc/SegDiff/logs/scripts/pannuke/FPN_0.2_aug/epoch=75-val_IoU=0.5236.ckpt
# CUDA_VISIBLE_DEVICES=4 python Mytest.py --root_path ../data/Pannuke --csv_path pannuke.csv --label_ratio 0.2 --model fpn --num_classes 6 --exp Pannuke/CoTraining



# ########pannuke2

# ### pannuke2 fpn
# CUDA_VISIBLE_DEVICES=6 python train_deep_co_training_2D.py --root_path ../data/Pannuke_2 --csv_path pannuke_2.csv --label_ratio 0.05 --model fpn --num_classes 6 --exp Pannuke_2/CoTraining --pretrained /data114_2/shaozc/SegDiff/logs/scripts/pannuke_2/FPN_0.05_aug/epoch=111-val_IoU=0.4295.ckpt
# CUDA_VISIBLE_DEVICES=6 python Mytest.py --root_path ../data/Pannuke_2 --csv_path pannuke_2.csv --label_ratio 0.05 --model fpn --num_classes 6 --exp Pannuke_2/CoTraining

# CUDA_VISIBLE_DEVICES=6 python train_deep_co_training_2D.py --root_path ../data/Pannuke_2 --csv_path pannuke_2.csv --label_ratio 0.1 --model fpn --num_classes 6 --exp Pannuke_2/CoTraining --pretrained /data114_2/shaozc/SegDiff/logs/scripts/pannuke_2/FPN_0.1_aug/epoch=88-val_IoU=0.4847.ckpt
# CUDA_VISIBLE_DEVICES=6 python Mytest.py --root_path ../data/Pannuke_2 --csv_path pannuke_2.csv --label_ratio 0.1 --model fpn --num_classes 6 --exp Pannuke_2/CoTraining

# CUDA_VISIBLE_DEVICES=6 python train_deep_co_training_2D.py --root_path ../data/Pannuke_2 --csv_path pannuke_2.csv --label_ratio 0.2 --model fpn --num_classes 6 --exp Pannuke_2/CoTraining --pretrained /data114_2/shaozc/SegDiff/logs/scripts/pannuke_2/FPN_0.2_aug/epoch=77-val_IoU=0.4999.ckpt
# CUDA_VISIBLE_DEVICES=6 python Mytest.py --root_path ../data/Pannuke_2 --csv_path pannuke_2.csv --label_ratio 0.2 --model fpn --num_classes 6 --exp Pannuke_2/CoTraining


# ########pannuke3

# ### pannuke3 fpn
# CUDA_VISIBLE_DEVICES=6 python train_deep_co_training_2D.py --root_path ../data/Pannuke_3 --csv_path pannuke_3.csv --label_ratio 0.05 --model fpn --num_classes 6 --exp Pannuke_3/CoTraining --pretrained /data114_2/shaozc/SegDiff/logs/scripts/pannuke_3/FPN_0.05_aug/epoch=151-val_IoU=0.4381.ckpt
# CUDA_VISIBLE_DEVICES=6 python Mytest.py --root_path ../data/Pannuke_3 --csv_path pannuke_3.csv --label_ratio 0.05 --model fpn --num_classes 6 --exp Pannuke_3/CoTraining

# CUDA_VISIBLE_DEVICES=6 python train_deep_co_training_2D.py --root_path ../data/Pannuke_3 --csv_path pannuke_3.csv --label_ratio 0.1 --model fpn --num_classes 6 --exp Pannuke_3/CoTraining --pretrained /data114_2/shaozc/SegDiff/logs/scripts/pannuke_3/FPN_0.1_aug/epoch=49-val_IoU=0.4613.ckpt
# CUDA_VISIBLE_DEVICES=6 python Mytest.py --root_path ../data/Pannuke_3 --csv_path pannuke_3.csv --label_ratio 0.1 --model fpn --num_classes 6 --exp Pannuke_3/CoTraining

# CUDA_VISIBLE_DEVICES=6 python train_deep_co_training_2D.py --root_path ../data/Pannuke_3 --csv_path pannuke_3.csv --label_ratio 0.2 --model fpn --num_classes 6 --exp Pannuke_3/CoTraining --pretrained /data114_2/shaozc/SegDiff/logs/scripts/pannuke_3/FPN_0.2_aug/epoch=42-val_IoU=0.5082.ckpt
# CUDA_VISIBLE_DEVICES=6 python Mytest.py --root_path ../data/Pannuke_3 --csv_path pannuke_3.csv --label_ratio 0.2 --model fpn --num_classes 6 --exp Pannuke_3/CoTraining



# ##########CoNIC
# # ########conic

# # # ### conic fpn
# # CUDA_VISIBLE_DEVICES=6 python train_deep_co_training_2D.py --root_path ../data/CoNIC --csv_path conic_42.csv --label_ratio 0.05 --model fpn --num_classes 7 --exp CoNIC/CoTraining --pretrained /data114_2/shaozc/SegDiff/logs/scripts/conic/FPN_0.05_aug/epoch=199-val_IoU=0.3814.ckpt
# CUDA_VISIBLE_DEVICES=6 python Mytest.py --root_path ../data/CoNIC --csv_path conic_42.csv --label_ratio 0.05 --model fpn --num_classes 7 --exp CoNIC/CoTraining

# # CUDA_VISIBLE_DEVICES=6 python train_deep_co_training_2D.py --root_path ../data/CoNIC --csv_path conic_42.csv --label_ratio 0.1 --model fpn --num_classes 7 --exp CoNIC/CoTraining --pretrained /data114_2/shaozc/SegDiff/logs/scripts/conic/FPN_0.1_aug/epoch=149-val_IoU=0.4027.ckpt
# CUDA_VISIBLE_DEVICES=6 python Mytest.py --root_path ../data/CoNIC --csv_path conic_42.csv --label_ratio 0.1 --model fpn --num_classes 7 --exp CoNIC/CoTraining

# # CUDA_VISIBLE_DEVICES=6 python train_deep_co_training_2D.py --root_path ../data/CoNIC --csv_path conic_42.csv --label_ratio 0.2 --model fpn --num_classes 7 --exp CoNIC/CoTraining --pretrained /data114_2/shaozc/SegDiff/logs/scripts/conic/FPN_0.2_aug/epoch=156-val_IoU=0.4348.ckpt
# CUDA_VISIBLE_DEVICES=6 python Mytest.py --root_path ../data/CoNIC --csv_path conic_42.csv --label_ratio 0.2 --model fpn --num_classes 7 --exp CoNIC/CoTraining


# # # ########conic_107

# # ### conic fpn
# CUDA_VISIBLE_DEVICES=6 python train_deep_co_training_2D.py --root_path ../data/CoNIC_107 --csv_path conic_107.csv --label_ratio 0.05 --model fpn --num_classes 7 --exp CoNIC_107/CoTraining --pretrained /data114_2/shaozc/SegDiff/logs/scripts/conic_107/FPN_0.05_aug/epoch=149-val_IoU=0.3862.ckpt
# CUDA_VISIBLE_DEVICES=6 python Mytest.py --root_path ../data/CoNIC_107 --csv_path conic_107.csv --label_ratio 0.05 --model fpn --num_classes 7 --exp CoNIC_107/CoTraining

# CUDA_VISIBLE_DEVICES=6 python train_deep_co_training_2D.py --root_path ../data/CoNIC_107 --csv_path conic_107.csv --label_ratio 0.1 --model fpn --num_classes 7 --exp CoNIC_107/CoTraining --pretrained /data114_2/shaozc/SegDiff/logs/scripts/conic_107/FPN_0.1_aug/epoch=167-val_IoU=0.4192.ckpt
# CUDA_VISIBLE_DEVICES=6 python Mytest.py --root_path ../data/CoNIC_107 --csv_path conic_107.csv --label_ratio 0.1 --model fpn --num_classes 7 --exp CoNIC_107/CoTraining

# CUDA_VISIBLE_DEVICES=6 python train_deep_co_training_2D.py --root_path ../data/CoNIC_107 --csv_path conic_107.csv --label_ratio 0.2 --model fpn --num_classes 7 --exp CoNIC_107/CoTraining --pretrained /data114_2/shaozc/SegDiff/logs/scripts/conic_107/FPN_0.2_aug/epoch=71-val_IoU=0.4269.ckpt
# CUDA_VISIBLE_DEVICES=6 python Mytest.py --root_path ../data/CoNIC_107 --csv_path conic_107.csv --label_ratio 0.2 --model fpn --num_classes 7 --exp CoNIC_107/CoTraining

# # ########conic_412

# # ### conic fpn
# CUDA_VISIBLE_DEVICES=6 python train_deep_co_training_2D.py --root_path ../data/CoNIC_412 --csv_path conic_412.csv --label_ratio 0.05 --model fpn --num_classes 7 --exp CoNIC_412/CoTraining --pretrained /data114_2/shaozc/SegDiff/logs/scripts/conic_412/FPN_0.05_aug/epoch=160-val_IoU=0.3814.ckpt
# CUDA_VISIBLE_DEVICES=6 python Mytest.py --root_path ../data/CoNIC_412 --csv_path conic_412.csv --label_ratio 0.05 --model fpn --num_classes 7 --exp CoNIC_412/CoTraining

# CUDA_VISIBLE_DEVICES=6 python train_deep_co_training_2D.py --root_path ../data/CoNIC_412 --csv_path conic_412.csv --label_ratio 0.1 --model fpn --num_classes 7 --exp CoNIC_412/CoTraining --pretrained /data114_2/shaozc/SegDiff/logs/scripts/conic_412/FPN_0.1_aug/epoch=149-val_IoU=0.4100.ckpt
# CUDA_VISIBLE_DEVICES=6 python Mytest.py --root_path ../data/CoNIC_412 --csv_path conic_412.csv --label_ratio 0.1 --model fpn --num_classes 7 --exp CoNIC_412/CoTraining

# CUDA_VISIBLE_DEVICES=6 python train_deep_co_training_2D.py --root_path ../data/CoNIC_412 --csv_path conic_412.csv --label_ratio 0.2 --model fpn --num_classes 7 --exp CoNIC_412/CoTraining --pretrained /data114_2/shaozc/SegDiff/logs/scripts/conic_412/FPN_0.2_aug/epoch=78-val_IoU=0.4277.ckpt
# CUDA_VISIBLE_DEVICES=6 python Mytest.py --root_path ../data/CoNIC_412 --csv_path conic_412.csv --label_ratio 0.2 --model fpn --num_classes 7 --exp CoNIC_412/CoTraining






parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/ACDC', help='Name of Experiment')
parser.add_argument('--csv_path', type=str,
                    default='pannuke_3.csv', help='Name of Experiment')
parser.add_argument('--label_ratio', type=float,  default=0.05, help='label_ratio')
parser.add_argument('--pretrained', type=str,
                    default='/data114_2/shaozc/SegDiff/logs/scripts/pannuke_3/PSPNet_0.2_aug/epoch=125-val_IoU=0.4824.ckpt', help='pretrained_path')
parser.add_argument('--exp', type=str,
                    default='Pannuke/CoTraining', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='unet', help='model_name')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=48,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[256, 256],
                    help='patch size of network input')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--num_classes', type=int,  default=4,
                    help='output channel of network')

# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=24,
                    help='labeled_batch_size per gpu')
parser.add_argument('--labeled_num', type=int, default=3,
                    help='labeled data')
# costs
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,
                    default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,
                    default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=200.0, help='consistency_rampup')
args = parser.parse_args()


def patients_to_slices(dataset, patiens_num):
    ref_dict = None
    if "ACDC" in dataset:
        ref_dict = {"3": 68, "7": 136,
                    "14": 256, "21": 396, "28": 512, "35": 664, "140": 1312}
    elif "Prostate":
        ref_dict = {"2": 27, "4": 53, "8": 120,
                    "12": 179, "16": 256, "21": 312, "42": 623}
    else:
        print("Error")
    return ref_dict[str(patiens_num)]


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes)

    #---->load the pretrained weight
    pretrained_dict = torch.load(args.pretrained, map_location='cpu')['model_state_dict']
    pretrained_dict = {k.replace('model.',''):v for k,v in pretrained_dict.items()}
    msg = model.load_state_dict(pretrained_dict, strict=False) #feature extractor do not need to load the weight
    print('model_image', msg)
    model.cuda()

    db_train = BaseDataSetsPro(base_dir=args.root_path, csv_dir=args.csv_path, split="train", transform=transforms.Compose([
        RandomGenerator(args.patch_size)
    ]))

    total_slices = len(db_train)
    labeled_slice = int(len(db_train) * args.label_ratio)
    print("Total silices is: {}, labeled slices is: {}".format(
        total_slices, labeled_slice))
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs, unlabeled_idxs, batch_size, batch_size-args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler,
                             num_workers=16, pin_memory=True, worker_init_fn=worker_init_fn)

    db_val = BaseDataSetsPro(base_dir=args.root_path, csv_dir=args.csv_path, split="val", num=args.label_ratio)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False,
                           num_workers=1)

    model.train()

    optimizer = optim.SGD(model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)

    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    early_stop = 0 ###2
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):

            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            unlabeled_volume_batch = volume_batch[args.labeled_bs:]

            outputs = model(volume_batch)
            outputs_soft = torch.softmax(outputs, dim=1)

            rot_times = random.randrange(0,4)

            rotated_unlabeled_volume_batch = torch.rot90(unlabeled_volume_batch, rot_times, [2,3])

            unlabeled_rot_outputs = model(rotated_unlabeled_volume_batch)
            unlabeled_rot_outputs_soft = torch.softmax(unlabeled_rot_outputs, dim=1)

            loss_ce = ce_loss(outputs[:args.labeled_bs],
                              label_batch[:][:args.labeled_bs].long())
            loss_dice = dice_loss(
                outputs_soft[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1))
            supervised_loss = 0.5 * (loss_dice + loss_ce)

            consistency_weight = get_current_consistency_weight(iter_num//150)

            consistency_loss = 0.5 * (torch.mean((unlabeled_rot_outputs_soft.detach() - torch.rot90(outputs_soft[args.labeled_bs:], rot_times, [2,3]))**2) + torch.mean((unlabeled_rot_outputs_soft - torch.rot90(outputs_soft[args.labeled_bs:].detach(), rot_times, [2,3]))**2))

            loss = supervised_loss + consistency_weight * consistency_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/loss_dice', loss_dice, iter_num)
            writer.add_scalar('info/consistency_loss',
                              consistency_loss, iter_num)
            writer.add_scalar('info/consistency_weight',
                              consistency_weight, iter_num)

            logging.info(
                'iteration %d : loss : %f, loss_ce: %f, loss_dice: %f' %
                (iter_num, loss.item(), loss_ce.item(), loss_dice.item()))

            if iter_num % 20 == 0:
                image = volume_batch[1, 0:1, :, :]
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(
                    outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction',
                                 outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

            if iter_num > 0 and iter_num % 200 == 0: ###6
                model.eval()

                max_probs = []
                target = []
                for i_batch, sampled_batch in enumerate(valloader):
                    model.eval()
                    logits = model(sampled_batch["image"].cuda())
                    max_probs.append(torch.topk(logits, 1, dim = 1)[1])
                    target.append(sampled_batch["label"])
                max_probs = torch.cat(max_probs, dim = 0)
                target = torch.cat(target, dim = 0)
                #---->
                metrics = dict()
                tp, fp, fn, tn = smp.metrics.get_stats(max_probs.squeeze().cpu(), target.squeeze().cpu().long(), mode='multiclass', num_classes=args.num_classes)

                iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="macro")
                f1_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction="macro")

                metrics['val_IoU'] = iou_score
                metrics['val_f1'] = f1_score


                writer.add_scalar('info/val_iou_score', iou_score, iter_num)
                writer.add_scalar('info/val_f1_score', f1_score, iter_num)

                if iou_score > best_performance:
                    best_performance = iou_score
                    save_mode_path = os.path.join(snapshot_path,
                                                    'iter_{}_iou_{}.pth'.format(
                                                        iter_num, np.round(best_performance.numpy(), 4)))
                    save_best = os.path.join(snapshot_path,
                                                '{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best)
                    early_stop = 0

                else:
                    early_stop = early_stop + 1

                logging.info(
                    'iteration %d : mean_iou_score : %f mean_f1_score : %f' % (iter_num, iou_score, f1_score))
                model.train()

            # if iter_num % 3000 == 0:
            #     save_mode_path = os.path.join(
            #         snapshot_path, 'iter_' + str(iter_num) + '.pth')
            #     torch.save(model.state_dict(), save_mode_path)
            #     logging.info("save model to {}".format(save_mode_path))

            if iter_num >= max_iterations or early_stop >= 5:
                break
        if iter_num >= max_iterations or early_stop >= 5:
            iterator.close()
            if early_stop >= 5:
                print('#########EARLY STOP###########')
            break
    writer.close()
    return "Training Finished!"


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = "../model/{}_{}_labeled/{}".format(
        args.exp, args.label_ratio, args.model)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code',
                    shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)
