#!/usr/bin/env python3
"""VideoVAEPlus pretraining for ultrasound videos.

Architecture: 2+1D ConvNet VAE with temporal compression + 3D PatchGAN.
  Encoder2plus1D -> TemporalEncoder1DCNN -> latent z
  Latent z -> TemporalDecoder1DCNN -> Decoder2plus1D -> reconstruction
  Loss: L1 + LPIPS + KL + 3D PatchGAN discriminator

Based on VideoVAEPlus (pretraining/VideoVAEPlus/) config_4z.yaml.

Usage:
  # 1. Generate annotation files (once):
  python pretraining/prepare_data.py \\
      --video_dirs /path/to/videos \\
      --data_root /path/to/dataset \\
      --output_dir pretraining/data

  # 2. Launch training (8 GPUs):
  torchrun --nproc_per_node=8 pretraining/run_pretrain_vae.py \\
      --data_root /path/to/dataset \\
      --data_path pretraining/data/us_videomae_train.txt \\
      --output_dir pretraining/output/vae_4z

  # 3. Convert checkpoint for buildmodel.py:
  python pretraining/convert_checkpoint.py \\
      --method vae \\
      --input pretraining/output/vae_4z/checkpoint-99.pth \\
      --output pretraining/output/vae_4z/encoder_checkpoint.pt
"""

import argparse
import datetime
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import warnings
warnings.filterwarnings("ignore")
# Ensure the parent directory is on sys.path
_project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

_pretrain_dir = os.path.dirname(os.path.abspath(__file__))
if _pretrain_dir not in sys.path:
    sys.path.insert(0, _pretrain_dir)

# Also add VideoVAEPlus for LPIPS/taming dependency
_vaeplus_dir = os.path.join(_pretrain_dir, "VideoVAEPlus")
if _vaeplus_dir not in sys.path:
    sys.path.insert(0, _vaeplus_dir)

from methods.vae_modeling_pretrain import VideoVAEPretrainModel, get_vae_config
from methods.vae_dataset import build_vae_pretraining_dataset
from methods.vae_engine import train_one_epoch
from methods.vae_utils import (
    NativeScalerWithGradNormCount,
    auto_load_model,
    cosine_scheduler,
    get_rank,
    get_world_size,
    init_distributed_mode,
    is_main_process,
    save_model,
    seed_worker,
    TensorboardLogger,
)


def get_args():
    parser = argparse.ArgumentParser("VideoVAEPlus pretraining on ultrasound videos", add_help=False)
    parser.add_argument("--batch_size", default=2, type=int, help="Per-GPU batch size")
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--save_ckpt_freq", default=1, type=int)

    # Model
    parser.add_argument("--embed_dim", default=4, type=int, help="VAE latent dim")
    parser.add_argument("--z_channels", default=4, type=int, help="Decoder input channels")
    parser.add_argument("--temporal_scale_factor", default=4, type=int,
                        help="Temporal compression factor (4x)")
    parser.add_argument("--disc_start", default=50001, type=int,
                        help="Global step to start discriminator training")
    parser.add_argument("--kl_weight", default=8.33e-8, type=float, help="KL divergence weight")
    parser.add_argument("--disc_weight", default=0.5, type=float, help="Discriminator loss weight")
    parser.add_argument("--input_size", default=224, type=int, help="Input spatial size")

    # ViT encoder (matches vit_giant_xformers / MAE)
    parser.add_argument("--encoder_embed_dim", default=1408, type=int)
    parser.add_argument("--encoder_depth", default=40, type=int)
    parser.add_argument("--encoder_num_heads", default=22, type=int)
    parser.add_argument("--drop_path", default=0.0, type=float)
    parser.add_argument("--with_cp", action="store_true", default=True,
                        help="Gradient checkpointing for ViT encoder")

    # Optimizer
    parser.add_argument("--opt", default="adamw", type=str)
    parser.add_argument("--opt_eps", default=1e-8, type=float)
    parser.add_argument("--opt_betas", default=[0.5, 0.9], type=float, nargs="+")
    parser.add_argument("--clip_grad", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--weight_decay_end", type=float, default=None)

    # LR
    parser.add_argument("--lr", type=float, default=1e-4, help="Base learning rate")
    parser.add_argument("--warmup_lr", type=float, default=1e-8)
    parser.add_argument("--min_lr", type=float, default=1e-7)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--warmup_steps", type=int, default=-1)

    # Dataset
    parser.add_argument("--data_path", default="pretraining/data/us_videomae_train.txt", type=str)
    parser.add_argument("--data_root", default="/home/wcz/workspace/DATASET", type=str)
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--sampling_rate", type=int, default=4)
    parser.add_argument("--num_segments", type=int, default=1)
    parser.add_argument("--temporal_jitter", action="store_true", default=False)

    # Output
    parser.add_argument("--output_dir", default="pretraining/output/vae_4z", type=str)
    parser.add_argument("--log_dir", default=None, type=str)

    # System
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--resume", default="")
    parser.add_argument("--auto_resume", action="store_true", default=True)
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--pin_mem", action="store_true", default=True)

    # Debug
    parser.add_argument("--debug", action="store_true", default=False)

    # DDP
    parser.add_argument("--world_size", default=1, type=int)
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument("--dist_url", default="env://")

    return parser.parse_args()


def param_groups_weight_decay(model, weight_decay=0.0):
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim == 1 or name.endswith(".bias"):
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
    ddconfig, ppconfig, lossconfig, vitconfig = get_vae_config(
        embed_dim=args.embed_dim,
        resolution=args.input_size,
        z_channels=args.z_channels,
        temporal_scale_factor=args.temporal_scale_factor,
        encoder_embed_dim=args.encoder_embed_dim,
        encoder_depth=args.encoder_depth,
        encoder_num_heads=args.encoder_num_heads,
    )
    lossconfig["disc_start"] = args.disc_start
    lossconfig["kl_weight"] = args.kl_weight
    lossconfig["disc_weight"] = args.disc_weight

    model = VideoVAEPretrainModel(
        ddconfig=ddconfig,
        ppconfig=ppconfig,
        lossconfig=lossconfig,
        embed_dim=args.embed_dim,
        use_quant_conv=False,
        img_size=args.input_size,
        num_frames=args.num_frames,
        encoder_embed_dim=args.encoder_embed_dim,
        encoder_depth=args.encoder_depth,
        encoder_num_heads=args.encoder_num_heads,
        encoder_mlp_ratio=vitconfig["mlp_ratio"],
        drop_path_rate=args.drop_path,
        with_cp=args.with_cp,
    )

    # ================================================================
    # Build dataset
    # ================================================================
    dataset_train = build_vae_pretraining_dataset(args)

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

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
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
    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % total_batch_size)
    print("Number of training steps = %d" % num_training_steps_per_epoch)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True
        )
        model_without_ddp = model.module

    # ================================================================
    # Optimizers (two: AE + Discriminator)
    # ================================================================
    # Separate AE params from discriminator params
    ae_params = []
    disc_params = []
    for name, param in model_without_ddp.named_parameters():
        if not param.requires_grad:
            continue
        if "loss.discriminator" in name:
            disc_params.append(param)
        else:
            ae_params.append(param)

    param_groups = param_groups_weight_decay(model_without_ddp, args.weight_decay)

    optimizer_ae = torch.optim.AdamW(
        param_groups,
        lr=args.lr,
        betas=tuple(args.opt_betas),
        eps=args.opt_eps,
        weight_decay=0.0,
    )
    optimizer_disc = torch.optim.AdamW(
        disc_params,
        lr=args.lr,
        betas=tuple(args.opt_betas),
        eps=args.opt_eps,
    )
    loss_scaler = NativeScalerWithGradNormCount()

    # ================================================================
    # Schedulers
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
        args.weight_decay, args.weight_decay_end, args.epochs, num_training_steps_per_epoch,
    )
    print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))

    # ================================================================
    # Debug mode overrides
    # ================================================================
    if args.debug:
        args.epochs = 2
        args.auto_resume = False
        debug_max_steps = 3
        print("[DEBUG] Overriding: epochs=2, max_steps=3, auto_resume=False")
    else:
        debug_max_steps = None

    # ================================================================
    # Auto-resume
    # ================================================================
    auto_load_model(
        args=args, model=model, model_without_ddp=model_without_ddp,
        optimizer=optimizer_ae, loss_scaler=loss_scaler,
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

    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch)

        train_stats = train_one_epoch(
            model, data_loader_train, optimizer_ae, optimizer_disc,
            device, epoch, loss_scaler,
            max_norm=args.clip_grad,
            log_writer=log_writer,
            start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values,
            wd_schedule_values=wd_schedule_values,
            log_file=log_fh,
            disc_start=args.disc_start,
            max_steps=debug_max_steps,
        )

        if args.output_dir:
            _epoch = epoch + 1
            if _epoch % args.save_ckpt_freq == 0 or _epoch == args.epochs:
                save_model(
                    args=args, epoch=epoch, model=model,
                    model_without_ddp=model_without_ddp,
                    optimizer=optimizer_ae, loss_scaler=loss_scaler,
                )
                # Also save ViT encoder in build_model()-compatible format
                if is_main_process():
                    encoder_state = model_without_ddp.get_encoder_state_dict()
                    encoder_path = os.path.join(args.output_dir, f"encoder_epoch_{_epoch}.pt")
                    torch.save(
                        {"encoder": encoder_state, "classifiers": []},
                        encoder_path,
                    )
                    print(f"Saved build_model-compatible encoder to {encoder_path}")

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
