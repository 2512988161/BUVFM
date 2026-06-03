"""
Stage 1 — Classification distillation: ViT-G teacher → MobileNetV3 student backbone.

Teacher processes a video clip [B, C, T=16, H=224, W=224] through ViT-G,
producing spatio-temporal features [B, 8, 14, 14, 1408].

Student processes each frame independently [B*T, C, H, W] through MobileNetV3
→ [B*T, 40, 14, 14] → project → [B*T, 1408, 14, 14] → temporal pool → [B, 8, 1408, 14, 14].

Loss: α·MSE + β·(1 − cosine_similarity)
"""
import warnings
warnings.filterwarnings("ignore")
import os
import sys
import argparse
import logging
import numpy as np
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

# Ensure project root is on sys.path before local imports
_VJEPA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _VJEPA_ROOT not in sys.path:
    sys.path.insert(0, _VJEPA_ROOT)

from mydataset import VideoFolderDataset
from utils import make_transforms

from distill.cls.buildmodel import (
    MobileNetV3Backbone,
    Student2TeacherProjector,
    TemporalAggregator,
    load_teacher,
    extract_teacher_features,
    save_distill_checkpoint,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='CLS Distillation: ViT-G → MobileNetV3')
    # Teacher
    p.add_argument('--teacher_ckpt', type=str, required=True,
                   help='Path to ViT-G checkpoint (from train.py output)')
    # Data
    p.add_argument('--video_dir', type=str, default='/home/lx/dataset/videos_train',
                   help='Video dataset root (class_0/, class_1/, ... subdirs)')
    # Output
    p.add_argument('--output_dir', type=str, default=None,
                   help='Output dir for weights/logs (default: distill/cls/output/distill/)')
    # Training
    p.add_argument('--batch_size', type=int, default=4,
                   help='Per-GPU batch size')
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight_decay', type=float, default=0.05)
    p.add_argument('--num_frames', type=int, default=16)
    p.add_argument('--frame_step', type=int, default=2)
    # Loss weights
    p.add_argument('--alpha', type=float, default=1.0,
                   help='MSE loss weight')
    p.add_argument('--beta', type=float, default=0.5,
                   help='Cosine-similarity loss weight')
    # Student
    p.add_argument('--model_name', type=str, default='mobilenetv3_small_075',
                   help='timm MobileNetV3 model name')
    # Misc
    p.add_argument('--num_workers', type=int, default=8)
    p.add_argument('--log_interval', type=int, default=10,
                   help='Log every N steps (per rank 0)')
    return p.parse_args()


# ---------------------------------------------------------------------------
# Distillation loss
# ---------------------------------------------------------------------------

def distill_loss(student_feat, teacher_feat, alpha=1.0, beta=0.5):
    """
    Args:
        student_feat: [B, T, C, H, W]  projected student features
        teacher_feat: [B, T, H, W, C]  teacher features (from extract_teacher_features)

    Returns (total_loss, mse_val, cosine_val).
    """
    teacher = teacher_feat.permute(0, 1, 4, 2, 3)  # [B, T, H, W, C] → [B, T, C, H, W]

    mse = F.mse_loss(student_feat, teacher)

    # Flatten all dims beyond batch for cosine
    s_flat = student_feat.flatten(1)
    t_flat = teacher.flatten(1)
    cosine = (1.0 - F.cosine_similarity(s_flat, t_flat, dim=-1)).mean()

    total = alpha * mse + beta * cosine
    return total, mse.detach(), cosine.detach()


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
    output_dir = args.output_dir or os.path.join(os.path.dirname(__file__), 'output/distill')
    os.makedirs(output_dir, exist_ok=True)

    if local_rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(output_dir, 'distill.log')),
                logging.StreamHandler(),
            ],
            force=True,
        )
    logger = logging.getLogger(__name__)

    # ---- Dataset ----
    transform = make_transforms(
        training=True, num_views_per_clip=1, crop_size=224,
        random_horizontal_flip=True, random_resize_scale=(0.3, 1.0),
        random_resize_aspect_ratio=(0.75, 1.3333),
    )

    dataset = VideoFolderDataset(
        root_dir=args.video_dir,
        frames_per_clip=args.num_frames,
        frame_step=args.frame_step,
        num_clips=1,
        transform=transform,
        random_clip_sampling=True,
    )
    sampler = DistributedSampler(dataset)
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler,
                        num_workers=args.num_workers, pin_memory=True, drop_last=True)

    # ---- Teacher ----
    if local_rank == 0:
        logger.info(f'Loading teacher from {args.teacher_ckpt}')
    teacher = load_teacher(args.teacher_ckpt, num_frames=args.num_frames, device='cuda')
    # teacher is already frozen and on the correct device via load_teacher → but we need DDP
    # For KD we only need inference, so DDP wrapping is unnecessary.
    # However, if we wrap it in DDP, we get one copy per GPU (which is what we want).
    # Actually, we don't need to wrap it — just move to the correct device.
    teacher = teacher.cuda(local_rank)

    # ---- Student ----
    backbone = MobileNetV3Backbone(
        model_name=args.model_name, num_classes=2,
    ).cuda(local_rank)

    projector = Student2TeacherProjector(
        in_channels=backbone.feature_channels, out_channels=1408,
    ).cuda(local_rank)

    temporal_agg = TemporalAggregator().cuda(local_rank)

    # DDP
    backbone = DDP(backbone, device_ids=[local_rank], find_unused_parameters=True)
    projector = DDP(projector, device_ids=[local_rank], find_unused_parameters=False)
    # temporal_agg has no parameters, so it doesn't need DDP wrapping

    # ---- Optimizer ----
    params = list(backbone.parameters()) + list(projector.parameters())
    optimizer = optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler('cuda')

    # ---- Training loop ----
    best_loss = float('inf')

    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        backbone.train()
        projector.train()

        epoch_loss = torch.zeros(3).cuda(local_rank)  # [total, mse, cosine]

        pbar = tqdm(loader, desc=f'Epoch {epoch:03d}', disable=(local_rank != 0))
        for step, batch in enumerate(pbar):
            clips, labels, _clip_indices = batch

            # Extract video tensor from nested clip structure.
            # clips is list-of-lists: [num_clips][num_views] → tensor([B, C, T, H, W])
            videos = clips[0][0].cuda(local_rank, non_blocking=True)  # [B, C, T, H, W]
            B, C, T, H, W = videos.shape

            # --- Teacher forward (no grad) ---
            teacher_feat = extract_teacher_features(teacher, videos)
            # [B, 8, 14, 14, 1408]

            # --- Student forward ---
            # Per-frame: [B, C, T, H, W] → [B*T, C, H, W]
            frames = videos.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda', dtype=torch.float16):
                feat = backbone(frames)                          # [B*T, 576, 14, 14]
                feat = projector(feat)                           # [B*T, 1408, 14, 14]
                feat = temporal_agg(feat, T=T)                   # [B, 8, 1408, 14, 14]
                loss, mse_val, cos_val = distill_loss(
                    feat, teacher_feat, args.alpha, args.beta,
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss[0] += loss.item()
            epoch_loss[1] += mse_val.item()
            epoch_loss[2] += cos_val.item()

            if local_rank == 0 and step % args.log_interval == 0:
                pbar.set_postfix(loss=f'{loss.item():.4f}',
                                 mse=f'{mse_val.item():.4f}',
                                 cos=f'{cos_val.item():.4f}')

        # ---- End-of-epoch sync ----
        dist.all_reduce(epoch_loss, op=dist.ReduceOp.SUM)

        global_total = epoch_loss[0] / (step + 1) / world_size
        global_mse = epoch_loss[1] / (step + 1) / world_size
        global_cos = epoch_loss[2] / (step + 1) / world_size

        if local_rank == 0:
            logger.info(
                f'Epoch {epoch+1:03d}/{args.epochs} | '
                f'Loss: {global_total:.6f} (MSE: {global_mse:.6f}, Cos: {global_cos:.6f})'
            )

            if global_total < best_loss:
                best_loss = global_total
                save_distill_checkpoint(
                    backbone.module, projector.module, temporal_agg,
                    optimizer, epoch,
                    {'loss': global_total, 'mse': global_mse, 'cosine': global_cos},
                    os.path.join(output_dir, 'best_distill.pt'),
                )
                logger.info(f'  → Saved best model (loss={best_loss:.6f})')

            # Periodic checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                save_distill_checkpoint(
                    backbone.module, projector.module, temporal_agg,
                    optimizer, epoch,
                    {'loss': global_total, 'mse': global_mse, 'cosine': global_cos},
                    os.path.join(output_dir, f'checkpoint_epoch_{epoch+1:03d}.pt'),
                )

    dist.destroy_process_group()


if __name__ == '__main__':
    main()
