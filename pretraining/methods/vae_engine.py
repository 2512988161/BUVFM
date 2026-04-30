"""Training engine for VideoVAEPlus pretraining.

Follows the mae_engine.py pattern but with two-optimizer setup (AE + Discriminator).
"""

import math
import sys
from typing import Iterable
import torch
from .vae_utils import MetricLogger, SmoothedValue, get_grad_norm_


def train_one_epoch(
    model: torch.nn.Module,
    data_loader: Iterable,
    optimizer_ae: torch.optim.Optimizer,
    optimizer_disc: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    max_norm: float = 0,
    log_writer=None,
    start_steps=None,
    lr_schedule_values=None,
    wd_schedule_values=None,
    log_file=None,
    disc_start: int = 50001,
):
    model.train()
    model_without_ddp = model.module if hasattr(model, 'module') else model
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)
    if log_file is not None:
        log_file.write(header + "\n")
        log_file.flush()
    print_freq = 20

    for step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header, log_file=log_file)):
        it = start_steps + step

        # Step-level LR / WD scheduling
        if lr_schedule_values is not None or wd_schedule_values is not None:
            for param_group in optimizer_ae.param_groups:
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group.get("lr_scale", 1.0)
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        global_step = it

        images = batch.to(device, non_blocking=True)

        # ---- AE (generator) step ----
        optimizer_ae.zero_grad()
        with torch.cuda.amp.autocast():
            reconstructions, posterior = model(images)

        loss_ae, log_ae = model_without_ddp.loss(
            images, reconstructions, posterior, 0, global_step,
            last_layer=model_without_ddp.get_last_layer(), split="train",
        )

        if not math.isfinite(loss_ae.item()):
            print(f"Loss is {loss_ae.item()}, stopping training")
            sys.exit(2)

        if loss_scaler is not None:
            is_second_order = hasattr(optimizer_ae, "is_second_order") and optimizer_ae.is_second_order
            grad_norm = loss_scaler(
                loss_ae, optimizer_ae, clip_grad=max_norm,
                parameters=model.parameters(), create_graph=is_second_order,
            )
            loss_scale_value = loss_scaler.state_dict()["scale"]
        else:
            loss_ae.backward()
            if max_norm is None:
                grad_norm = get_grad_norm_(model.parameters())
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer_ae.step()
            loss_scale_value = 0

        # ---- Discriminator step ----
        optimizer_disc.zero_grad()
        if global_step >= disc_start:
            loss_disc, log_disc = model_without_ddp.loss(
                images, reconstructions.detach(), posterior, 1, global_step,
                last_layer=model_without_ddp.get_last_layer(), split="train",
            )
            loss_disc.backward()
            clip_value = 0.01
            torch.nn.utils.clip_grad_norm_(model_without_ddp.loss.discriminator.parameters(), clip_value)
            optimizer_disc.step()
            disc_loss_value = loss_disc.item()
        else:
            disc_loss_value = 0.0

        torch.cuda.synchronize()

        loss_value = loss_ae.item()
        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_scale=loss_scale_value)
        metric_logger.update(disc_loss=disc_loss_value)

        min_lr = 10.0
        max_lr = 0.0
        for group in optimizer_ae.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])
        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)

        weight_decay_value = None
        for group in optimizer_ae.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        # Log
        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")
            for k, v in log_ae.items():
                if isinstance(v, torch.Tensor) and v.numel() == 1:
                    log_writer.update(v.item(), head=k)
            log_writer.set_step()

    metric_logger.synchronize_between_processes()
    avg_msg = "Averaged stats: " + str(metric_logger)
    print(avg_msg)
    if log_file is not None:
        log_file.write(avg_msg + "\n")
        log_file.flush()
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
