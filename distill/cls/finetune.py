"""
Stage 2 — Classification fine-tuning.

Loads a distilled MobileNetV3 backbone from Stage 1, then fine-tunes the full
model (backbone + head) on a 2D 2-class image dataset (ImageFolder layout).
"""

import os
import sys
import argparse
import logging

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import datasets, transforms
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import numpy as np

_VJEPA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _VJEPA_ROOT not in sys.path:
    sys.path.insert(0, _VJEPA_ROOT)

from distill.cls.buildmodel import MobileNetV3Backbone, load_distilled_backbone


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='CLS Fine-tuning on 2D images')
    # Checkpoint
    p.add_argument('--distilled_ckpt', type=str, default='',
                   help='Path to Stage-1 distillation checkpoint (best_distill.pt). '
                        'If empty or missing, train from scratch.')
    # Data
    p.add_argument('--train_dir', type=str,
                   default='/home/lx/dataset/2cls_classification/train',
                   help='2D image training set (ImageFolder layout)')
    p.add_argument('--val_dir', type=str,
                   default='/home/lx/dataset/2cls_classification/val',
                   help='2D image validation set (ImageFolder layout)')
    # Output
    p.add_argument('--output_dir', type=str, default=None,
                   help='Output dir (default: distill/cls/output/)')
    # Model
    p.add_argument('--num_classes', type=int, default=2)
    p.add_argument('--model_name', type=str, default='mobilenetv3_small_075',
                   help='timm MobileNetV3 model name')
    p.add_argument('--freeze_backbone', action='store_true',
                   help='Freeze distilled backbone layers, train only head')
    # Training
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--label_smoothing', type=float, default=0.0)
    p.add_argument('--warmup_epochs', type=int, default=0,
                   help='Linear warmup epochs (0 to disable)')
    p.add_argument('--randaug_magnitude', type=int, default=0,
                   help='RandAugment magnitude (0 to disable)')
    # Misc
    p.add_argument('--num_workers', type=int, default=8)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def build_train_transforms(crop_size=224, randaug_magnitude=0):
    aug = []
    aug.append(transforms.RandomResizedCrop(crop_size, scale=(0.5, 1.0)))
    aug.append(transforms.RandomHorizontalFlip())
    if randaug_magnitude > 0:
        aug.append(transforms.RandAugment(num_ops=2, magnitude=randaug_magnitude))
    aug.append(transforms.ToTensor())
    aug.append(transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD))
    if randaug_magnitude > 0:
        aug.append(transforms.RandomErasing(p=0.25))
    return transforms.Compose(aug)

def build_val_transforms(crop_size=224):
    return transforms.Compose([
        transforms.Resize(int(crop_size * 256 / 224)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # ---- DDP init ----
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)

    # ---- Logging ----
    output_dir = args.output_dir or os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(output_dir, exist_ok=True)
    if local_rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(output_dir, 'finetune.log')),
                logging.StreamHandler(),
            ],
            force=True,
        )
    logger = logging.getLogger(__name__)

    # ---- Datasets ----
    train_set = datasets.ImageFolder(
        root=args.train_dir,
        transform=build_train_transforms(randaug_magnitude=args.randaug_magnitude),
    )
    val_set = datasets.ImageFolder(root=args.val_dir,
                                   transform=build_val_transforms())

    train_sampler = DistributedSampler(train_set)
    val_sampler   = DistributedSampler(val_set, shuffle=False)
    train_loader  = DataLoader(train_set, batch_size=args.batch_size, sampler=train_sampler,
                               num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader    = DataLoader(val_set,   batch_size=args.batch_size, sampler=val_sampler,
                               num_workers=args.num_workers, pin_memory=True, drop_last=False)

    # ---- Model ----
    if local_rank == 0:
        logger.info(f'Loading distilled backbone from {args.distilled_ckpt}')

    backbone, meta, load_info = load_distilled_backbone(
        args.distilled_ckpt,
        model_name=args.model_name,
        num_classes=args.num_classes,
    )
    if local_rank == 0:
        logger.info(
            f'Loading distilled backbone from {args.distilled_ckpt}: '
            f'matched {load_info["matched"]}, extra {load_info["extra"]}, '
            f'missing {load_info["missing"]}'
        )
    backbone = backbone.cuda(local_rank)

    if args.freeze_backbone:
        for p in backbone.conv_stem.parameters():   p.requires_grad = False
        for p in backbone.bn1.parameters():         p.requires_grad = False
        for p in backbone.blocks_early.parameters(): p.requires_grad = False
        if local_rank == 0:
            logger.info('Froze backbone (conv_stem + blocks_early), training head only')

    # Wrap in DDP — forward_full gives logits
    model = DDP(backbone, device_ids=[local_rank], find_unused_parameters=False)

    # ---- Loss & Optimizer ----
    labels = train_set.targets
    label_counts = np.bincount(labels, minlength=args.num_classes)
    weights = 1.0 / torch.tensor(label_counts, dtype=torch.float)
    weights = weights / weights.sum() * args.num_classes
    weights = weights.cuda(local_rank)

    if local_rank == 0:
        logger.info(f'Class counts: {label_counts.tolist()}, weights: {weights.tolist()}')

    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=args.label_smoothing)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.warmup_epochs > 0:
        def lr_lambda(epoch):
            if epoch < args.warmup_epochs:
                return (epoch + 1) / args.warmup_epochs
            progress = (epoch - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
            return 0.5 * (1.0 + np.cos(np.pi * progress))
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    scaler = torch.amp.GradScaler('cuda')

    # ---- Training ----
    best_acc = 0.0

    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        model.train()

        if args.freeze_backbone:
            model.module.conv_stem.eval()
            model.module.bn1.eval()
            model.module.blocks_early.eval()

        train_metrics = torch.zeros(3).cuda(local_rank)  # loss, correct, total

        pbar = tqdm(train_loader, desc=f'Epoch {epoch:03d}', disable=(local_rank != 0))
        for imgs, targets in pbar:
            imgs = imgs.cuda(local_rank, non_blocking=True)
            targets = targets.cuda(local_rank, non_blocking=True)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda', dtype=torch.float16):
                logits = model.module.forward_full(imgs)
                loss = criterion(logits, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            preds = logits.argmax(1)
            train_metrics[0] += loss.item() * imgs.size(0)
            train_metrics[1] += (preds == targets).sum().item()
            train_metrics[2] += imgs.size(0)

            pbar.set_postfix(loss=f'{loss.item():.4f}')

        scheduler.step()

        # ---- Validation ----
        model.eval()
        val_metrics = torch.zeros(3).cuda(local_rank)
        all_preds, all_labels = [], []

        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs = imgs.cuda(local_rank, non_blocking=True)
                targets = targets.cuda(local_rank, non_blocking=True)

                with torch.amp.autocast('cuda', dtype=torch.float16):
                    logits = model.module.forward_full(imgs)
                    loss = criterion(logits, targets)

                preds = logits.argmax(1)
                val_metrics[0] += loss.item() * imgs.size(0)
                val_metrics[1] += (preds == targets).sum().item()
                val_metrics[2] += imgs.size(0)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(targets.cpu().numpy())

        # ---- Sync ----
        dist.all_reduce(train_metrics, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_metrics,   op=dist.ReduceOp.SUM)

        train_loss = train_metrics[0] / train_metrics[2]
        train_acc  = 100. * train_metrics[1] / train_metrics[2]
        val_loss   = val_metrics[0] / val_metrics[2]
        val_acc    = 100. * val_metrics[1] / val_metrics[2]

        # Gather all preds/labels for confusion matrix
        all_preds_list  = [None] * world_size
        all_labels_list = [None] * world_size
        dist.all_gather_object(all_preds_list, all_preds)
        dist.all_gather_object(all_labels_list, all_labels)

        if local_rank == 0:
            logger.info(
                f'Epoch {epoch+1:03d}/{args.epochs} | '
                f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | '
                f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%'
            )

            flat_preds  = [p for sub in all_preds_list for p in sub]
            flat_labels = [l for sub in all_labels_list for l in sub]
            cm = confusion_matrix(flat_labels, flat_preds)
            logger.info(f'Confusion Matrix:\n{cm}')

            if val_acc > best_acc:
                best_acc = val_acc
                acc_str = f'{best_acc:.2f}'.replace('.', '')
                torch.save({
                    'model_state_dict': model.module.state_dict(),
                    'epoch': epoch,
                    'val_acc': val_acc,
                    'meta': meta,
                }, os.path.join(output_dir, f'best_finetune_{acc_str}.pt'))
                logger.info(f'  → Saved best model ({best_acc:.2f}%)')

            # Periodic
            if (epoch + 1) % 20 == 0:
                torch.save({
                    'model_state_dict': model.module.state_dict(),
                    'epoch': epoch,
                    'val_acc': val_acc,
                }, os.path.join(output_dir, f'checkpoint_epoch_{epoch+1:03d}.pt'))

    dist.destroy_process_group()


if __name__ == '__main__':
    main()
