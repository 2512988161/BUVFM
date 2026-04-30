#!/usr/bin/env python3
"""Standalone VideoMAEv2-style MAE pretraining for VJEPA2 ViT-g on ultrasound videos.

Reference: VideoMAEv2 (https://github.com/OpenGVLab/VideoMAEv2)
  Architecture: VJEPA ViT-g encoder (RoPE + SwiGLU) + shallow decoder
  Masking:      Tube masking (90%) encoder, Running-cell masking (50%) decoder
  Loss:         Per-patch normalized MSE
  Optimizer:    AdamW (beta=0.9, 0.95) with cosine LR/WD schedule

Usage:
  # 1. Generate annotation files (once):
  python pretraining/prepare_data.py \\
      --video_dirs /home/wcz/workspace/DATASET/us_foundation_model_dataset_videos_videos \\
                   /home/wcz/workspace/DATASET/us_foundation_model_dataset_img_videos \\
      --data_root /home/wcz/workspace/DATASET \\
      --output_dir pretraining/data

  # 2. Launch training (8 GPUs):
  torchrun --nproc_per_node=8 pretraining/run_pretrain_mae.py \\
      --data_root /home/wcz/workspace/DATASET \\
      --data_path pretraining/data/us_videomae_train.txt \\
      --output_dir pretraining/output/mae_vitg

  # 3. Convert checkpoint for buildmodel.py:
  python pretraining/convert_checkpoint.py \\
      --method videomae \\
      --input pretraining/output/mae_vitg/checkpoint-299.pth \\
      --output pretraining/output/mae_vitg/encoder_checkpoint.pt
"""

import argparse
import datetime
import json
import os
import random
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import warnings
warnings.filterwarnings("ignore")
# Ensure the parent directory (vjepa root) is on sys.path so that
# 'import src.models.vision_transformer' and 'import methods.*' both work.
_project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Also add pretraining/ so "import methods.*" resolves
_pretrain_dir = os.path.dirname(os.path.abspath(__file__))
if _pretrain_dir not in sys.path:
    sys.path.insert(0, _pretrain_dir)

from methods.mae_modeling_pretrain import MAEPretrainModel
from methods.mae_dataset import build_pretraining_dataset
from methods.mae_engine import train_one_epoch
from methods.mae_utils import (
    NativeScalerWithGradNormCount,
    auto_load_model,
    cosine_scheduler,
    get_rank,
    get_world_size,
    init_distributed_mode,
    is_main_process,
    multiple_pretrain_samples_collate,
    save_model,
    seed_worker,
    TensorboardLogger,
)


def get_args():
    parser = argparse.ArgumentParser(
        "VJEPA2 MAE pretraining on ultrasound videos", add_help=False
    )
    parser.add_argument("--batch_size", default=4, type=int, help="Per-GPU batch size")
    parser.add_argument("--epochs", default=300, type=int)
    parser.add_argument("--save_ckpt_freq", default=1, type=int)

    # Model
    parser.add_argument("--tubelet_size", type=int, default=2)
    parser.add_argument("--with_checkpoint", action="store_true", default=True,
                        help="Use gradient checkpointing for encoder")
    parser.add_argument("--decoder_depth", default=4, type=int, help="Decoder depth")
    parser.add_argument("--mask_type", default="tube", choices=["random", "tube"],
                        help="Encoder mask strategy")
    parser.add_argument("--decoder_mask_type", default="run_cell",
                        choices=["random", "run_cell"], help="Decoder mask strategy")
    parser.add_argument("--mask_ratio", default=0.90, type=float, help="Encoder mask ratio")
    parser.add_argument("--decoder_mask_ratio", default=0.50, type=float,
                        help="Decoder mask ratio (0 = no decoder masking)")
    parser.add_argument("--input_size", default=224, type=int, help="Input spatial size")
    parser.add_argument("--drop_path", type=float, default=0.0, help="Drop path rate")
    parser.add_argument("--normlize_target", default=True, type=bool,
                        help="Normalize target patches per-patch")

    # Optimizer
    parser.add_argument("--opt", default="adamw", type=str)
    parser.add_argument("--opt_eps", default=1e-8, type=float)
    parser.add_argument("--opt_betas", default=[0.9, 0.95], type=float, nargs="+")
    parser.add_argument("--clip_grad", type=float, default=0.02, help="Gradient clipping (0.02 for Giant)")
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--weight_decay_end", type=float, default=None)

    # LR
    parser.add_argument("--lr", type=float, default=6e-4, help="Base LR (scaled by total_batch/256)")
    parser.add_argument("--warmup_lr", type=float, default=1e-6)
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--warmup_epochs", type=int, default=30)
    parser.add_argument("--warmup_steps", type=int, default=-1)

    # Dataset
    parser.add_argument("--data_path", default="pretraining/data/us_videomae_train.txt", type=str)
    parser.add_argument("--data_root", default="/home/wcz/workspace/DATASET", type=str)
    parser.add_argument("--fname_tmpl", default="img_{:05}.jpg", type=str)
    parser.add_argument("--imagenet_default_mean_and_std", default=True, action="store_true")
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--sampling_rate", type=int, default=4)
    parser.add_argument("--num_sample", type=int, default=4,
                        help="Repeated augmentation samples per video")

    # Output
    parser.add_argument("--output_dir", default="pretraining/output/mae_vitg", type=str)
    parser.add_argument("--log_dir", default=None, type=str,
                        help="Log directory for TensorBoard logs (default: no TensorBoard)")

    # System
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--resume", default="")
    parser.add_argument("--auto_resume", action="store_true", default=True)
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument("--pin_mem", action="store_true", default=True)

    # DDP
    parser.add_argument("--world_size", default=1, type=int)
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument("--dist_url", default="env://")

    return parser.parse_args()


def param_groups_weight_decay(model, weight_decay=0.05):
    """Separate params into weight-decay and no-weight-decay groups.

    Replicates VideoMAEv2's get_parameter_groups behaviour for AdamW:
    biases + 1D params (LayerNorm scales) + model.no_weight_decay() get no decay.
    """
    decay, no_decay = [], []
    skip_keywords = set()
    if hasattr(model, "no_weight_decay"):
        skip_keywords = set(model.no_weight_decay())

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if (param.ndim == 1 or name.endswith(".bias")
                or name in skip_keywords
                or any(kw in name for kw in skip_keywords)):
            no_decay.append(param)
        else:
            decay.append(param)

    return [
        {"params": decay, "weight_decay": weight_decay, "lr_scale": 1.0},
        {"params": no_decay, "weight_decay": 0.0, "lr_scale": 1.0},
    ]


def main(args):
    init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)
    seed = args.seed + get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    # ================================================================
    # Build model
    # ================================================================
    model = MAEPretrainModel(
        img_size=args.input_size,
        patch_size=16,
        num_frames=args.num_frames,
        tubelet_size=args.tubelet_size,
        in_chans=3,
        encoder_embed_dim=1408,
        encoder_depth=40,
        encoder_num_heads=22,
        encoder_mlp_ratio=48 / 11,
        decoder_embed_dim=512,
        decoder_depth=args.decoder_depth,
        decoder_num_heads=8,
        drop_path_rate=args.drop_path,
        use_rope=True,
        uniform_power=True,
        with_cp=args.with_checkpoint,
    )

    # Store patch_size for engine
    patch_size = 16
    args.patch_size = patch_size
    args.window_size = (
        args.num_frames // args.tubelet_size,
        args.input_size // patch_size,
        args.input_size // patch_size,
    )

    # ================================================================
    # Build dataset
    # ================================================================
    dataset_train = build_pretraining_dataset(args)

    num_tasks = get_world_size()
    global_rank = get_rank()
    total_batch_size = args.batch_size * num_tasks
    num_training_steps_per_epoch = len(dataset_train) // total_batch_size

    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    print("Sampler_train = %s" % str(sampler_train))

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = TensorboardLogger(log_dir=args.log_dir)
    else:
        log_writer = None

    if args.num_sample > 1:
        collate_func = partial(multiple_pretrain_samples_collate, fold=False)
    else:
        collate_func = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        collate_fn=collate_func,
        worker_init_fn=seed_worker,
        persistent_workers=True,
    )

    # ================================================================
    # Prepare model (device, DDP, parameter count)
    # ================================================================
    model.to(device)
    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Model = %s" % str(model_without_ddp))
    print("Number of params: %.2f M" % (n_parameters / 1e6))

    # Linear LR scaling
    args.lr = args.lr * total_batch_size / 256
    args.min_lr = args.min_lr * total_batch_size / 256
    args.warmup_lr = args.warmup_lr * total_batch_size / 256
    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % total_batch_size)
    print("Number of training steps = %d" % num_training_steps_per_epoch)
    print("Number of training examples per epoch = %d"
          % (total_batch_size * num_training_steps_per_epoch))

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=False
        )
        model_without_ddp = model.module

    # ================================================================
    # Optimizer
    # ================================================================
    param_groups = param_groups_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=args.lr,
        betas=tuple(args.opt_betas),
        eps=args.opt_eps,
        weight_decay=0.0,  # handled per-group
    )
    loss_scaler = NativeScalerWithGradNormCount()

    # ================================================================
    # Schedulers (step-level, precomputed arrays)
    # ================================================================
    print("Use step level LR & WD scheduler!")
    lr_schedule_values = cosine_scheduler(
        args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs,
        start_warmup_value=args.warmup_lr,
        warmup_steps=args.warmup_steps,
    )
    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    wd_schedule_values = cosine_scheduler(
        args.weight_decay, args.weight_decay_end, args.epochs,
        num_training_steps_per_epoch,
    )
    print("Max WD = %.7f, Min WD = %.7f"
          % (max(wd_schedule_values), min(wd_schedule_values)))

    # ================================================================
    # Auto-resume
    # ================================================================
    auto_load_model(
        args=args,
        model=model,
        model_without_ddp=model_without_ddp,
        optimizer=optimizer,
        loss_scaler=loss_scaler,
    )
    torch.cuda.empty_cache()

    # ================================================================
    # Training loop
    # ================================================================
    print("Start training for %d epochs" % args.epochs)
    log_dir = args.log_dir if args.log_dir else args.output_dir
    log_fh = None
    if log_dir and is_main_process():
        os.makedirs(log_dir, exist_ok=True)
        log_fh = open(os.path.join(log_dir, "training_log.txt"), "a")
        log_fh.write(f"Start training for {args.epochs} epochs\n")
        log_fh.flush()
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch)

        train_stats = train_one_epoch(
            model,
            data_loader_train,
            optimizer,
            device,
            epoch,
            loss_scaler,
            args.clip_grad,
            patch_size=patch_size,
            normlize_target=args.normlize_target,
            tubelet_size=args.tubelet_size,
            log_writer=log_writer,
            start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values,
            wd_schedule_values=wd_schedule_values,
            log_file=log_fh,
        )

        if args.output_dir:
            _epoch = epoch + 1
            if _epoch % args.save_ckpt_freq == 0 or _epoch == args.epochs:
                save_model(
                    args=args,
                    epoch=epoch,
                    model=model,
                    model_without_ddp=model_without_ddp,
                    optimizer=optimizer,
                    loss_scaler=loss_scaler,
                )

        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            "epoch": epoch,
            "n_parameters": n_parameters,
        }

        if args.output_dir and is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    end_msg = "Training time {}".format(total_time_str)
    print(end_msg)
    if log_fh is not None:
        log_fh.write(end_msg + "\n")
        log_fh.close()


if __name__ == "__main__":
    opts = get_args()
    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    main(opts)
