import os
import torch
import torch.nn.functional as F
import torch.distributed as dist
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from buildmodel import build_model
from mydataset import InferenceVideoDataset
from utils import make_transforms


def make_eval_dataloader(root_path, batch_size, **kwargs):
    """构建专门用于推理的 DataLoader（DDP 版本）"""
    transform = make_transforms(
        training=False,
        num_views_per_clip=kwargs.get("num_views_per_segment", 1),
        crop_size=kwargs.get("img_size", 224),
        speckle_noise_ratio=kwargs.get("speckle_noise_ratio", 0),
    )

    dataset = InferenceVideoDataset(
        root_dir=root_path,
        frames_per_clip=kwargs.get("frames_per_clip", 16),
        frame_step=kwargs.get("frame_step", 2),
        num_clips=kwargs.get("num_segments", 1),
        transform=transform,
        shared_transform=None,
        random_clip_sampling=False,
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
    all_labels = []
    all_paths = []

    with torch.no_grad():
        loader = tqdm(val_loader, desc=f"Inference Progress [Rank {rank}]") if rank == 0 else val_loader
        for i, data in enumerate(loader):
            clips, labels, clip_indices, paths = data
            labels = labels.to(device)

            clips_gpu = [[dij.to(device, non_blocking=True) for dij in di] for di in clips]

            with torch.amp.autocast('cuda', dtype=torch.float16):
                features = model(clips_gpu)
                logits_list = [classifier(f) for f in features]
                probs = sum([F.softmax(l, dim=1) for l in logits_list]) / len(logits_list)

            all_probs.append(probs.cpu())
            all_labels.append(labels.cpu())
            all_paths.extend(paths)

    # --- 收集本卡结果 ---
    local_probs = torch.cat(all_probs, dim=0)   # [N_local, 3]
    local_labels = torch.cat(all_labels, dim=0)  # [N_local]
    local_size = torch.tensor([local_probs.shape[0]], device=device, dtype=torch.long)

    # 收集所有卡的数据量
    sizes = [torch.zeros(1, dtype=torch.long, device=device) for _ in range(world_size)]
    dist.all_gather(sizes, local_size)

    # 收集 probs 和 labels（先 padding 到最大 size）
    max_size = max(s.item() for s in sizes)
    pad_probs = torch.zeros(max_size, 3, device=device)
    pad_labels = torch.zeros(max_size, dtype=torch.long, device=device)
    pad_probs[:local_probs.shape[0]] = local_probs.to(device)
    pad_labels[:local_labels.shape[0]] = local_labels.to(device)

    gathered_probs = [torch.zeros(max_size, 3, device=device) for _ in range(world_size)]
    gathered_labels = [torch.zeros(max_size, dtype=torch.long, device=device) for _ in range(world_size)]
    dist.all_gather(gathered_probs, pad_probs)
    dist.all_gather(gathered_labels, pad_labels)

    # Move back to CPU for post-processing
    gathered_probs = [p.cpu() for p in gathered_probs]
    gathered_labels = [l.cpu() for l in gathered_labels]

    # 收集 paths（字符串列表）
    gathered_paths = [None for _ in range(world_size)]
    dist.all_gather_object(gathered_paths, all_paths)

    # --- 只在 rank 0 整理并保存结果 ---
    if rank == 0:
        results = []
        for r in range(world_size):
            n = sizes[r].item()
            probs_r = gathered_probs[r][:n]
            labels_r = gathered_labels[r][:n]
            paths_r = gathered_paths[r]

            for j in range(n):
                video_name = os.path.basename(paths_r[j])
                results.append({
                    'video_name': video_name,
                    'label': labels_r[j].item(),
                    'p0': probs_r[j][0].item(),
                    'p1': probs_r[j][1].item(),
                    'p2': probs_r[j][2].item()
                })

        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False)
        print(f"\n=> 结果已保存至: {output_csv}")

        y_true = df['label'].values
        y_pred = df[['p0', 'p1', 'p2']].values.argmax(axis=1)

        acc = accuracy_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)

        print("\n" + "=" * 40)
        print(f"OVERALL ACCURACY: {acc * 100:.2f}%")
        print("=" * 40)

        print("\nCONFUSION MATRIX:")
        print("真实类别(行) \\ 预测类别(列)")
        print(pd.DataFrame(cm,
                           index=[f'True_{i}' for i in range(3)],
                           columns=[f'Pred_{i}' for i in range(3)]))
        print("=" * 40)


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
                        default="./output/maee1_huigu.csv",
                        help="OUT CSV PATH")
    parser.add_argument("--val_dir", type=str, nargs="+",
                        default=["/home/lx/alg/videos_val", "/home/lx/alg/videos_test"],
                        help="Validation video directories")
    parser.add_argument('--model_name', type=str, default='vit_giant_xformers',
                        help="模型名，目前支持vit_giant_xformers、vit_large、vit_huge")

    parser.add_argument("--restore_true", action="store_true",
                        help="Run robust evaluation with speckle noise ratios from 0.05 to 0.95")

    opt = parser.parse_args()

    args = {
        "batch_size": 16,
        "img_size": 224,
        "frames_per_clip": 16,
        "frame_step": 2,
        "num_segments": 1,
        "num_views_per_segment": 1,
        "num_workers": 8,
        "model_name": opt.model_name,
    }

    if opt.restore_true:
        ratios = np.arange(0.05, 1.0, 0.05)
        out_dir = "./output/newinternal"
        if rank == 0:
            os.makedirs(out_dir, exist_ok=True)

        for ratio in ratios:
            ratio = round(ratio, 2)
            if rank == 0:
                print(f"\n{'=' * 60}")
                print(f"=> Robust inference with speckle_noise_ratio={ratio}")
                print(f"{'=' * 60}")

            args["speckle_noise_ratio"] = ratio
            output_csv = os.path.join(out_dir, f"robust_{ratio}.csv")
            infer(opt.checkpoint, opt.val_dir, output_csv, args, device)
    else:
        output_csv = opt.output
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        infer(opt.checkpoint, opt.val_dir, output_csv, args, device)

    dist.destroy_process_group()
