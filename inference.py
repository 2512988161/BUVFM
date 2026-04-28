import os
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

# 导入你现有的模块 (请确保模块名与你的文件名对应)
from buildmodel import build_model  
from mydataset import InferenceVideoDataset
from utils import make_transforms 

def make_eval_dataloader(root_path, batch_size, **kwargs):
    """构建专门用于推理的 DataLoader"""
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

    # 单卡推理，不需要 DistributedSampler，直接顺序加载
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,       # 必须为 False 保证顺序
        num_workers=kwargs.get("num_workers", 8),
        pin_memory=True,
        drop_last=False,     # 必须为 False 保证不漏数据
    )
    return data_loader

# ==========================================
# 2. 主推理流程
# ==========================================
def infer(checkpoint_path, val_dir, output_csv, args):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # --- 加载模型 ---
    print("=> 正在加载模型...")
    model, classifier = build_model(
        checkpoint_path=checkpoint_path,
        resolution=args["img_size"],
        frames_per_clip=args["frames_per_clip"],
        num_classes=3,
        num_heads=16,
        num_probe_blocks=1
    )
    
    model = model.to(device)
    classifier = classifier.to(device)
    model.eval()
    classifier.eval()

    # --- 加载数据 ---
    val_loader = make_eval_dataloader(val_dir, **args)
    print(f"=> 准备推理，共 {len(val_loader.dataset)} 个视频，{len(val_loader)} 个 Batch。")

    # --- 推理循环 ---
    results =[]
    
    with torch.no_grad():
        for i, data in enumerate(tqdm(val_loader, desc="Inference Progress")):
            # 解析数据
            clips, labels, clip_indices, paths = data
            labels = labels.to(device)
            
            # 将视频数据放到 GPU (处理嵌套的 List[List[Tensor]])
            clips_gpu = [[dij.to(device, non_blocking=True) for dij in di] for di in clips]
            
            # 混合精度加速 (如果你原本训练用了 bfloat16 或 float16)
            with torch.amp.autocast('cuda', dtype=torch.float16):
                # 1. 提取特征 (输出特征列表，对应多个 views)
                features = model(clips_gpu)
                
                # 2. 分类器前向传播
                logits_list = [classifier(f) for f in features]
                
                # 3. 结果集成 (求平均概率)
                probs = sum([F.softmax(l, dim=1) for l in logits_list]) / len(logits_list)
            
            # 收集结果
            probs_cpu = probs.cpu().numpy()
            labels_cpu = labels.cpu().numpy()
            
            for j in range(len(paths)):
                video_name = os.path.basename(paths[j])  # 仅保留文件名
                results.append({
                    'video_name': video_name,
                    'label': labels_cpu[j],
                    'p0': probs_cpu[j][0],
                    'p1': probs_cpu[j][1],
                    'p2': probs_cpu[j][2]
                })

    # --- 整理并保存结果 ---
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"\n=> 结果已保存至: {output_csv}")

    # --- 计算准确率和混淆矩阵 ---
    y_true = df['label'].values
    # 预测类别为 p0, p1, p2 中概率最大的索引
    y_pred = df[['p0', 'p1', 'p2']].values.argmax(axis=1)

    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    print("\n" + "="*40)
    print(f"OVERALL ACCURACY: {acc * 100:.2f}%")
    print("="*40)
    
    print("\nCONFUSION MATRIX:")
    print("真实类别(行) \\ 预测类别(列)")
    print(pd.DataFrame(cm, 
                       index=[f'True_{i}' for i in range(3)], 
                       columns=[f'Pred_{i}' for i in range(3)]))
    print("="*40)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="VJEPA Video Inference")
    parser.add_argument("--checkpoint", type=str,
                        default="/home/lx/baselines/vjepa/ckpts/vjepa_full/best_vjepa_model9720.pt",
                        help="Path to model checkpoint")
    parser.add_argument("--val_dir", type=str, nargs="+",
                        default= ["/home/lx/alg/videos_val"],
                        help="Validation video directories")

    parser.add_argument("--restore_true", action="store_true",
                        help="Run robust evaluation with speckle noise ratios from 0.05 to 0.95")

    opt = parser.parse_args()
    # checkpoint_path = "/home/lx/alg/baselines/vjepa/ckpts/vjepa_full/best_vjepa_model.pt"  # 替换为你的实际权重路径
    # # val_dir = ["/home/lx/alg/videos_val","/home/lx/alg/videos_test"]
    # val_dir = ["/home/lx/alg/videos_val"]
    # 保持与训练一致的参数
    args = {
        "batch_size": 16,
        "img_size": 224,
        "frames_per_clip": 16,
        "frame_step": 2,
        "num_segments": 1,
        "num_views_per_segment": 1,
        "num_workers": 8,
    }

    if opt.restore_true:
        ratios = np.arange(0.05, 1.0, 0.05)
        out_dir = "./output/robust"
        os.makedirs(out_dir, exist_ok=True)

        for ratio in ratios:
            ratio = round(ratio, 2)
            print(f"\n{'='*60}")
            print(f"=> Robust inference with speckle_noise_ratio={ratio}")
            print(f"{'='*60}")

            args["speckle_noise_ratio"] = ratio
            output_csv = os.path.join(out_dir, f"inference_result_{ratio}.csv")
            infer(opt.checkpoint, opt.val_dir, output_csv, args)
    else:
        output_csv = "./output/inference_results.csv"
        infer(opt.checkpoint, opt.val_dir, output_csv, args)
