import os
import torch
import torch.nn.functional as F
import torch.distributed as dist
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

from buildmodel import build_model
from mydataset import InferenceVideoDataset
from utils import make_transforms
import warnings
warnings.filterwarnings("ignore")

def make_eval_dataloader(root_path, batch_size, exclude_names=None, **kwargs):
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

    if exclude_names and len(exclude_names) > 0:
        keep_indices = [i for i, s in enumerate(dataset.samples)
                        if os.path.basename(s) not in exclude_names]
        dataset = Subset(dataset, keep_indices)

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

    # --- 读取已处理视频（支持断点续跑） ---
    completed_videos = set()
    if rank == 0:
        # 主输出 CSV
        if os.path.exists(output_csv):
            try:
                existing = pd.read_csv(output_csv)
                completed_videos.update(existing['video_name'].tolist())
            except Exception:
                pass
        # 各 rank 的临时 CSV（上次崩溃未合并的）
        for r in range(world_size):
            tmp_csv = output_csv.replace('.csv', f'_tmp_rank{r}.csv')
            if os.path.exists(tmp_csv):
                try:
                    tmp_df = pd.read_csv(tmp_csv)
                    completed_videos.update(tmp_df['video_name'].tolist())
                except Exception:
                    pass
        if completed_videos:
            print(f"=> 发现 {len(completed_videos)} 个已处理视频，将跳过")

    # 广播已处理集合到所有卡
    obj_list = [list(completed_videos)]
    dist.broadcast_object_list(obj_list, src=0)
    completed_videos = set(obj_list[0])

    # --- 加载数据（自动跳过已完成视频） ---
    val_loader = make_eval_dataloader(val_dir, exclude_names=completed_videos, **args)
    if rank == 0:
        print(f"=> 准备推理，剩余 {len(val_loader.dataset)} 个视频待处理，"
              f"每卡 {len(val_loader)} 个 Batch。")

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

    # --- 临时 CSV 路径（本卡增量写入） ---
    tmp_csv = output_csv.replace('.csv', f'_tmp_rank{rank}.csv')

    # --- 推理循环 ---

    with torch.no_grad():
        loader = tqdm(val_loader, desc=f"Inference [Rank {rank}]") if rank == 0 else val_loader
        for i, data in enumerate(loader):
            clips, labels, clip_indices, paths = data
            labels = labels.to(device)

            clips_gpu = [[dij.to(device, non_blocking=True) for dij in di] for di in clips]

            with torch.amp.autocast('cuda', dtype=torch.float16):
                features = model(clips_gpu)
                logits_list = [classifier(f) for f in features]
                probs = sum([F.softmax(l, dim=1) for l in logits_list]) / len(logits_list)

            probs_cpu = probs.cpu()
            labels_cpu = labels.cpu()

            # 即时写入本卡临时 CSV（每个 batch 追加）
            rows = []
            for j, p in enumerate(paths):
                video_name = os.path.basename(p)
                rows.append({
                    'video_name': video_name,
                    'label': labels_cpu[j].item(),
                    'p0': probs_cpu[j][0].item(),
                    'p1': probs_cpu[j][1].item(),
                    'p2': probs_cpu[j][2].item()
                })
            batch_df = pd.DataFrame(rows)
            batch_df.to_csv(tmp_csv, mode='a',
                            header=(i == 0), index=False)

    # --- barrier 保证所有卡都写完 ---
    dist.barrier()

    # --- rank 0 合并所有临时 CSV ---
    if rank == 0:
        _merge_temp_csvs(output_csv, world_size)
        _print_metrics(output_csv)


def _merge_temp_csvs(output_csv, world_size):
    """合并所有 rank 的临时 CSV + 已有主 CSV 到最终输出"""
    dfs = []

    # 已有主 CSV（断面续跑时保留的）
    if os.path.exists(output_csv):
        try:
            dfs.append(pd.read_csv(output_csv))
        except Exception:
            pass

    for r in range(world_size):
        tmp_csv = output_csv.replace('.csv', f'_tmp_rank{r}.csv')
        if os.path.exists(tmp_csv):
            try:
                dfs.append(pd.read_csv(tmp_csv))
            except Exception:
                pass

    if not dfs:
        print("=> 没有新结果需要合并。")
        return

    df = pd.concat(dfs, ignore_index=True)
    # 按 video_name 去重，保留最后出现的（即最新的临时 CSV 结果）
    df = df.drop_duplicates(subset='video_name', keep='last')
    df.to_csv(output_csv, index=False)
    print(f"\n=> 结果已保存至: {output_csv}，共 {len(df)} 条")

    # 清理临时 CSV
    for r in range(world_size):
        tmp_csv = output_csv.replace('.csv', f'_tmp_rank{r}.csv')
        if os.path.exists(tmp_csv):
            os.remove(tmp_csv)


def _print_metrics(output_csv):
    """打印混淆矩阵和准确率"""
    df = pd.read_csv(output_csv)
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
        out_dir = "./output/0506internal/"
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
