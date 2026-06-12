import os
import torch
import torch.nn.functional as F
import torch.distributed as dist
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from buildmodel import build_model
from mydataset import SimpleVideoDataset
from utils import make_transforms


def make_eval_dataloader(root_path, batch_size, **kwargs):
    """构建用于推理的 DataLoader（DDP 版本）"""
    transform = make_transforms(
        training=False,
        num_views_per_clip=kwargs.get("num_views_per_segment", 1),
        crop_size=kwargs.get("img_size", 224),
        speckle_noise_ratio=kwargs.get("speckle_noise_ratio", 0),
    )

    dataset = SimpleVideoDataset(
        root_dir=root_path,
        frames_per_clip=kwargs.get("frames_per_clip", 16),
        frame_step=kwargs.get("frame_step", 2),
        num_clips=kwargs.get("num_segments", 1),
        transform=transform,
    )

    sampler = DistributedSampler(
        dataset,
        shuffle=False,
        drop_last=False,
    )

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=kwargs.get("num_workers", 8),
        pin_memory=True,
        drop_last=False,
    )
    return data_loader


# ==========================================
# 2. 主推理流程（假设 dist 已初始化）
# ==========================================
def infer(checkpoint_path, val_dir, output_csv, args, device):

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # --- 加载模型 ---
    if rank == 0:
        print("=> 正在加载模型...")
    model, classifier = build_model(
        checkpoint_path=checkpoint_path,
        resolution=args["img_size"],
        frames_per_clip=args["frames_per_clip"],
        num_classes=3,
        num_heads=16,
        num_probe_blocks=1,
        model_name=args["model_name"],
    )

    model = model.to(device)
    classifier = classifier.to(device)
    model.eval()
    classifier.eval()

    # --- 加载数据 ---
    val_loader = make_eval_dataloader(val_dir, **args)
    if rank == 0:
        print(f"=> 准备推理，共 {len(val_loader.dataset)} 个视频，{len(val_loader)} 个 Batch（每卡）。")

    # --- 推理循环 ---
    all_probs = []
    all_paths = []

    with torch.no_grad():
        loader = tqdm(val_loader, desc=f"Inference Progress [Rank {rank}]") if rank == 0 else val_loader
        for i, data in enumerate(loader):
            clips, paths = data

            clips_gpu = [[dij.to(device, non_blocking=True) for dij in di] for di in clips]

            with torch.amp.autocast('cuda', dtype=torch.float16):
                features = model(clips_gpu)
                logits_list = [classifier(f) for f in features]
                probs = sum([F.softmax(l, dim=1) for l in logits_list]) / len(logits_list)

            all_probs.append(probs.cpu())
            all_paths.extend(paths)

    # --- 收集本卡结果 ---
    local_probs = torch.cat(all_probs, dim=0)  # [N_local, 3]
    local_size = torch.tensor([local_probs.shape[0]], device=device, dtype=torch.long)

    # 收集所有卡的数据量
    sizes = [torch.zeros(1, dtype=torch.long, device=device) for _ in range(world_size)]
    dist.all_gather(sizes, local_size)

    # 收集 probs（先 padding 到最大 size）
    max_size = max(s.item() for s in sizes)
    pad_probs = torch.zeros(max_size, 3, device=device)
    pad_probs[:local_probs.shape[0]] = local_probs.to(device)

    gathered_probs = [torch.zeros(max_size, 3, device=device) for _ in range(world_size)]
    dist.all_gather(gathered_probs, pad_probs)

    # Move back to CPU for post-processing
    gathered_probs = [p.cpu() for p in gathered_probs]

    # 收集 paths（字符串列表）
    gathered_paths = [None for _ in range(world_size)]
    dist.all_gather_object(gathered_paths, all_paths)

    # --- 只在 rank 0 整理并保存结果 ---
    if rank == 0:
        results = []
        for r in range(world_size):
            n = sizes[r].item()
            probs_r = gathered_probs[r][:n]
            paths_r = gathered_paths[r]

            for j in range(n):
                video_name = os.path.relpath(paths_r[j], val_dir)
                results.append({
                    'video_name': video_name,
                    'p0': probs_r[j][0].item(),
                    'p1': probs_r[j][1].item(),
                    'p2': probs_r[j][2].item(),
                })

        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False)
        print(f"\n=> 结果已保存至: {output_csv}")
        print(df.head())


if __name__ == "__main__":
    import argparse

    # --- DDP 初始化（全局一次） ---
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    parser = argparse.ArgumentParser(description="VJEPA Video Inference (DDP)")
    parser.add_argument("--checkpoint", type=str,
                        default="/home/lx/baselines/vjepa/ckpts/vjepa_full/best_vjepa_model9720.pt",
                        help="Path to model checkpoint")
    parser.add_argument("--output", type=str,
                        default="./output/test.csv",
                        help="OUT CSV PATH")
    parser.add_argument("--val_dir", type=str,
                        default="/home/wcz/workspace/DATASET/ALL_high_risk",
                        help="Validation video directory (recursive scan)")
    parser.add_argument('--model_name', type=str, default='vit_giant_xformers',
                        help="模型名，目前支持vit_giant_xformers、vit_large、vit_huge")

    parser.add_argument("--frame_step", type=int, default=2,
                        help="Step between sampled frames")
    parser.add_argument("--frames_per_clip", type=int, default=16,
                        help="Frames sampled per clip")
    parser.add_argument("--num_segments", type=int, default=1)
    parser.add_argument("--num_views_per_segment", type=int, default=1)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--speckle_noise_ratio", type=float, default=0)

    args = parser.parse_args()
    args = vars(args)

    infer(
        checkpoint_path=args["checkpoint"],
        val_dir=args["val_dir"],
        output_csv=args["output"],
        args=args,
        device=device,
    )

    dist.destroy_process_group()
