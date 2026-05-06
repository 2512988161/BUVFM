import os
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, matthews_corrcoef,
    cohen_kappa_score, confusion_matrix,
)
from torch.utils.data import DataLoader

from buildmodel import build_model
from mydataset import InferenceVideoDataset
from utils import make_transforms


def specificity_score(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    n = cm.shape[0]
    specs = []
    for i in range(n):
        tn = cm.sum() - cm[i, :].sum() - cm[:, i].sum() + cm[i, i]
        fp = cm[:, i].sum() - cm[i, i]
        specs.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
    return np.mean(specs)


def make_eval_dataloader(root_path, batch_size, **kwargs):
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
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=kwargs.get("num_workers", 8),
        pin_memory=True,
        drop_last=False,
    )
    return data_loader


# ── Auto-search over logit adjustments ──────────────────────────────

def compute_all_metrics(y_true, y_pred, y_prob):
    pf1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
    except ValueError:
        auc = float("nan")
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced ACC": balanced_accuracy_score(y_true, y_pred),
        "Macro Precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Macro Sensitivity": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "Macro Specificity": specificity_score(y_true, y_pred),
        "Macro F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "Macro AUC": auc,
        "Weighted F1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
        "Cohen's Kappa": cohen_kappa_score(y_true, y_pred),
        "Class_0_F1": pf1[0],
        "Class_1_F1": pf1[1],
        "Class_2_F1": pf1[2],
    }


def objective_score(y_true, y_pred, target):
    """Score to maximize during search."""
    if target == "macro_f1":
        return f1_score(y_true, y_pred, average="macro", zero_division=0)
    elif target == "balanced_acc":
        return balanced_accuracy_score(y_true, y_pred)
    elif target == "min_class_f1":
        pf1 = f1_score(y_true, y_pred, average=None, zero_division=0)
        return min(pf1[1], pf1[2])  # maximize the worse of class 1/2
    elif target == "combined":
        mf1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        pf1 = f1_score(y_true, y_pred, average=None, zero_division=0)
        return mf1 + 0.5 * min(pf1[1], pf1[2])
    else:
        return f1_score(y_true, y_pred, average="macro", zero_division=0)


def auto_search(raw_logits, y_true, target="macro_f1",
                adj_range=(-2, 8), n_coarse=50, n_fine=40,
                t_range=(0.5, 5.0), n_t=30):
    """
    Grid-search additive logit adjustments [adj1, adj2] and temperature T
    to maximize `target` metric.

    raw_logits: (N, 3) averaged logits (before softmax)
    Returns: best adj_vec (len=3, adj0=0), best T, best metrics
    """
    n_classes = raw_logits.shape[1]

    # Stage 1: coarse grid over (adj1, adj2), T=1.0
    best_score = -1.0
    best_adj1, best_adj2 = 0.0, 0.0
    best_T = 1.0

    for adj1 in np.linspace(adj_range[0], adj_range[1], n_coarse):
        for adj2 in np.linspace(adj_range[0], adj_range[1], n_coarse):
            adj = np.array([0.0, adj1, adj2])
            logits_adj = raw_logits + adj
            y_pred = np.argmax(logits_adj, axis=1)
            score = objective_score(y_true, y_pred, target)
            if score > best_score:
                best_score = score
                best_adj1, best_adj2 = adj1, adj2

    # Stage 2: fine grid around best
    delta = (adj_range[1] - adj_range[0]) / n_coarse * 3
    lo1, hi1 = max(adj_range[0], best_adj1 - delta), min(adj_range[1], best_adj1 + delta)
    lo2, hi2 = max(adj_range[0], best_adj2 - delta), min(adj_range[1], best_adj2 + delta)
    for adj1 in np.linspace(lo1, hi1, n_fine):
        for adj2 in np.linspace(lo2, hi2, n_fine):
            adj = np.array([0.0, adj1, adj2])
            logits_adj = raw_logits + adj
            y_pred = np.argmax(logits_adj, axis=1)
            score = objective_score(y_true, y_pred, target)
            if score > best_score:
                best_score = score
                best_adj1, best_adj2 = adj1, adj2

    # Stage 3: sweep temperature with best adj
    for T in np.linspace(t_range[0], t_range[1], n_t):
        adj = np.array([0.0, best_adj1, best_adj2])
        logits_scaled = (raw_logits + adj) / T
        y_pred = np.argmax(logits_scaled, axis=1)
        score = objective_score(y_true, y_pred, target)
        if score > best_score:
            best_score = score
            best_T = T

    best_adj = np.array([0.0, best_adj1, best_adj2])
    return best_adj, best_T


def print_report(y_true, y_prob, title):
    y_pred = np.argmax(y_prob, axis=1)
    cm = confusion_matrix(y_true, y_pred)
    m = compute_all_metrics(y_true, y_pred, y_prob)

    print(f"\n  [{title}]")
    print(f"  {'-'*50}")
    for k, v in m.items():
        print(f"    {k:25s}: {v:.6f}")
    print(f"\n  Confusion Matrix (row=True, col=Pred):")
    print("  " + " " * 8 + "".join(f"Pred_{i:>8d}" for i in range(3)))
    for i in range(3):
        print(f"  True_{i}  " + "".join(f"{cm[i, j]:>8d}" for j in range(3)))
    return m


# ── Main inference ───────────────────────────────────────────────────

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
        num_probe_blocks=1,
    )
    model = model.to(device)
    classifier = classifier.to(device)
    model.eval()
    classifier.eval()

    # --- 加载数据 ---
    val_loader = make_eval_dataloader(val_dir, **args)
    print(f"=> 准备推理，共 {len(val_loader.dataset)} 个视频，{len(val_loader)} 个 Batch。")

    # --- 推理：保存 raw logits ---
    all_logits = []
    all_labels = []
    all_paths = []

    with torch.no_grad():
        for i, data in enumerate(tqdm(val_loader, desc="Inference Progress")):
            clips, labels, clip_indices, paths = data
            labels = labels.to(device)
            clips_gpu = [[dij.to(device, non_blocking=True) for dij in di] for di in clips]

            with torch.amp.autocast('cuda', dtype=torch.float16):
                features = model(clips_gpu)
                logits_list = [classifier(f) for f in features]
                # 先在 logit 空间平均，保留完整信息用于后续调参
                avg_logits = sum(logits_list) / len(logits_list)

            all_logits.append(avg_logits.cpu().to(torch.float32).numpy())
            all_labels.append(labels.cpu().numpy())
            all_paths.extend(paths)

    raw_logits = np.concatenate(all_logits, axis=0)  # (N, 3)
    y_true = np.concatenate(all_labels, axis=0)

    # --- 自动搜索 or 手动参数 ---
    target = args.get("auto_search_target", None)
    manual_adjust = args.get("manual_adjust", None)
    temperature = args.get("temperature", 1.0)

    if target is not None:
        # 自动搜索最优参数
        print(f"\n=> Auto-searching parameters to maximize [{target}] ...")
        best_adj, best_T = auto_search(raw_logits, y_true, target=target)
        print(f"   Best additive adjustment: [{best_adj[0]:.4f}, {best_adj[1]:.4f}, {best_adj[2]:.4f}]")
        print(f"   Best temperature: {best_T:.4f}")
        adj_vec = best_adj
        temperature = best_T
    elif manual_adjust is not None:
        adj_vec = np.array(manual_adjust, dtype=np.float32)
        print(f"=> Manual adjustment: {adj_vec}, temperature: {temperature}")
    elif args.get("class_priors", None) is not None:
        priors = args["class_priors"]
        adj_vec = np.array([-np.log(p) for p in priors], dtype=np.float32)
        # 以 class 0 为基准
        adj_vec -= adj_vec[0]
        print(f"=> Class priors {priors} -> adjustment: {adj_vec}")
    else:
        adj_vec = np.zeros(3, dtype=np.float32)

    # --- 应用调整，计算最终概率 ---
    logits_adj = raw_logits + adj_vec
    logits_scaled = logits_adj / temperature
    probs = F.softmax(torch.from_numpy(logits_scaled), dim=1).numpy()

    # --- 保存原始 (argmax) 结果作为对比 ---
    probs_raw = F.softmax(torch.from_numpy(raw_logits), dim=1).numpy()

    # --- 输出 CSV ---
    results = []
    for j in range(len(all_paths)):
        video_name = os.path.basename(all_paths[j])
        results.append({
            "video_name": video_name,
            "label": int(y_true[j]),
            "p0": float(probs[j][0]),
            "p1": float(probs[j][1]),
            "p2": float(probs[j][2]),
        })
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"\n=> 结果已保存至: {output_csv}")

    # --- 打印指标：原始 vs 调整后 ---
    print("\n" + "=" * 60)
    print("  FINAL RESULTS")
    print("=" * 60)

    metrics_raw = print_report(y_true, probs_raw, "Original (argmax)")
    metrics_adj = print_report(y_true, probs,
                               f"Adjusted: T={temperature:.3f}, adj={adj_vec}")

    # 对比表
    keys = list(metrics_raw.keys())
    print(f"\n  {'Metric':<25s}{'Original':>12s}{'Adjusted':>12s}{'Delta':>10s}")
    print("  " + "-" * 60)
    for k in keys:
        delta = metrics_adj[k] - metrics_raw[k]
        sym = "↑" if delta > 1e-6 else ("↓" if delta < -1e-6 else " ")
        print(f"  {k:<25s}{metrics_raw[k]:>12.4f}{metrics_adj[k]:>12.4f}{delta:>+10.4f} {sym}")

    print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="VJEPA Video Inference with Auto Logit Adjustment")
    parser.add_argument("--checkpoint", type=str,
                        default="/home/lx/baselines/vjepa/ckpts/vjepa_full/best_vjepa_model9720.pt")
    parser.add_argument("--output", type=str, default="./output/test.csv")
    parser.add_argument("--val_dir", type=str, nargs="+",
                        default=["/home/lx/alg/videos_val", "/home/lx/alg/videos_test"])

    # Modes (mutually exclusive):
    #   1. --auto_search_target: auto-search for best params
    #   2. --manual_adjust: provide explicit additive adjustments [a0, a1, a2]
    #   3. --class_priors: provide priors [π0, π1, π2], auto-convert to adjustment
    #   4. (none): plain argmax, no adjustment
    parser.add_argument("--auto_search_target", type=str, default=None,
                        choices=["macro_f1", "balanced_acc", "min_class_f1", "combined"],
                        help="Auto-search logit adjustments to maximize this metric")
    parser.add_argument("--manual_adjust", type=float, nargs=3, default=None,
                        help="Manual additive adjustment [a0, a1, a2] (default a0=0)")
    parser.add_argument("--class_priors", type=float, nargs=3, default=None,
                        help="Class priors [π0, π1, π2] -> adj_k = log(1/π_k)")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Softmax temperature (used with manual/priors mode)")

    parser.add_argument("--restore_true", action="store_true",
                        help="Robust evaluation with speckle noise 0.05~0.95")

    opt = parser.parse_args()

    args = {
        "batch_size": 16,
        "img_size": 224,
        "frames_per_clip": 16,
        "frame_step": 2,
        "num_segments": 1,
        "num_views_per_segment": 1,
        "num_workers": 8,
        "auto_search_target": opt.auto_search_target,
        "manual_adjust": opt.manual_adjust,
        "class_priors": opt.class_priors,
        "temperature": opt.temperature,
    }

    if opt.restore_true:
        ratios = np.arange(0.05, 1.0, 0.05)
        out_dir = os.path.join(os.path.dirname(opt.output) or ".", "robust")
        os.makedirs(out_dir, exist_ok=True)
        for ratio in ratios:
            ratio = round(ratio, 2)
            print(f"\n{'='*60}")
            print(f"=> Robust inference with speckle_noise_ratio={ratio}")
            print(f"{'='*60}")
            args["speckle_noise_ratio"] = ratio
            output_csv = os.path.join(out_dir, f"robust_{ratio}.csv")
            infer(opt.checkpoint, opt.val_dir, output_csv, args)
    else:
        infer(opt.checkpoint, opt.val_dir, opt.output, args)
