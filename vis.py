import os
import cv2
import torch
import torch.nn.functional as F
import numpy as np
from decord import VideoReader, cpu
from buildmodel import build_model
from utils import make_transforms


def visualize_gradcam(
    video_path,
    checkpoint_path="./ckpts/vjepa_full/best_vjepa_model9720.pt",
    output_dir="./output",
    img_size=224,
    frames_per_clip=16,
    frame_step=2,
    num_classes=3,
    num_heads=16,
    num_probe_blocks=1,
    target_class=None,
    alpha=0.5,
):
    """
    对单个视频使用 Grad-CAM 可视化热力图，并叠加回视频帧上输出。

    Args:
        video_path: 输入视频路径
        checkpoint_path: 模型权重路径
        output_dir: 输出目录
        img_size: 输入分辨率
        frames_per_clip: 每个clip的帧数
        frame_step: 帧采样步长
        num_classes: 分类类别数
        num_heads: 分类器注意力头数
        num_probe_blocks: 分类器probe block数
        target_class: Grad-CAM目标类别 (None=使用预测类别)
        alpha: 热力图叠加透明度
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 1. 加载模型 ---
    print(f"=> 加载权重: {checkpoint_path}")
    model, classifier = build_model(
        checkpoint_path=checkpoint_path,
        resolution=img_size,
        frames_per_clip=frames_per_clip,
        num_classes=num_classes,
        num_heads=num_heads,
        num_probe_blocks=num_probe_blocks,
    )
    model = model.to(device).eval()
    classifier = classifier.to(device).eval()

    # --- 2. 读取视频 (与 test.py 相同的采样逻辑) ---
    print(f"=> 读取视频: {video_path}")
    vr = VideoReader(video_path, num_threads=-1, ctx=cpu(0))
    total_frames = len(vr)

    if total_frames < frames_per_clip * frame_step:
        indices = np.linspace(0, total_frames - 1, num=frames_per_clip).astype(np.int64)
    else:
        clip_len = frames_per_clip * frame_step
        start_idx = (total_frames - clip_len) // 2
        indices = np.arange(start_idx, start_idx + clip_len, frame_step)[:frames_per_clip]

    buffer = vr.get_batch(indices).asnumpy()  # (T, H, W, C) uint8
    print(f"=> 采样 {len(indices)} 帧, 原始视频总帧数: {total_frames}")

    # --- 3. Transform (与 test.py 一致) ---
    transform = make_transforms(training=False, num_views_per_clip=1, crop_size=img_size)
    clips = transform(buffer)  # List[Tensor(C, T, H, W)]

    # 构造输入格式: [[view_tensor(B, C, T, H, W)]]
    clips_input = [[c.unsqueeze(0).to(device) for c in clips]]

    # --- 4. 前向推理获取特征 (no_grad 节省显存) ---
    with torch.no_grad():
        features = model(clips_input)

    # 取出特征, detach 后启用梯度 (仅分类器反向传播需要)
    feat = features[0].float().detach().requires_grad_(True)  # (1, N, D)

    # --- 5. 分类器前向 + 反向传播 ---
    logits = classifier(feat)  # (1, num_classes)
    pred_class = logits.argmax(dim=1).item()
    target = target_class if target_class is not None else pred_class

    classifier.zero_grad()
    logits[0, target].backward()

    # --- 6. 计算 Grad-CAM ---
    activations = feat.detach()       # (1, N, D)
    gradients = feat.grad.detach()    # (1, N, D)

    # 通道权重: 对 token 维度做全局平均池化
    weights = gradients.mean(dim=1)   # (1, D)

    # 加权求和
    cam = (weights.unsqueeze(1) * activations).sum(dim=-1)  # (1, N)
    cam = F.relu(cam).squeeze(0)     # (N,)

    # 重塑为 (T_tokens, H_patches, W_patches)
    T_tokens = frames_per_clip // 2   # tubelet_size=2 → 8
    H_patches = img_size // 16        # patch_size=16 → 14
    W_patches = img_size // 16
    cam = cam.cpu().numpy().reshape(T_tokens, H_patches, W_patches)

    # 归一化到 [0, 1]
    cam_min, cam_max = cam.min(), cam.max()
    if cam_max - cam_min > 0:
        cam = (cam - cam_min) / (cam_max - cam_min)

    # --- 7. 将热力图叠加到原始帧上 ---
    original_frames = vr.get_batch(indices).asnumpy()  # (T, H, W, C) uint8

    # 与 transform 一致的 resize 参数
    short_side = int(img_size * 256 / 224)  # 256

    probs = F.softmax(logits.detach(), dim=1)
    prob_text = "  ".join([f"C{c}:{probs[0,c]:.3f}" for c in range(num_classes)])

    overlay_frames = []
    original_crop_frames = []

    for frame_idx in range(len(indices)):
        frame = original_frames[frame_idx].copy()
        h, w = frame.shape[:2]

        # Resize short side (与 transform 一致)
        if h < w:
            new_h, new_w = short_side, int(w * short_side / h)
        else:
            new_h, new_w = int(h * short_side / w), short_side
        frame_resized = cv2.resize(frame, (new_w, new_h))

        # CenterCrop
        start_h = (new_h - img_size) // 2
        start_w = (new_w - img_size) // 2
        frame_crop = frame_resized[start_h:start_h + img_size, start_w:start_w + img_size]

        original_crop_frames.append(frame_crop.copy())

        # 该帧对应的 temporal token 热力图 (tubelet_size=2, 每2帧共享1个token)
        token_idx = min(frame_idx // 2, T_tokens - 1)
        heatmap = cam[token_idx]  # (14, 14)

        # 上采样到 img_size × img_size
        heatmap_up = cv2.resize(heatmap, (img_size, img_size), interpolation=cv2.INTER_CUBIC)
        heatmap_uint8 = np.uint8(255 * heatmap_up)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

        # 叠加
        overlay = cv2.addWeighted(frame_crop, 1 - alpha, heatmap_color, alpha, 0)
        overlay_frames.append(overlay)

    # --- 8. 拼接左右对比帧: 原始 | 热力图 ---
    comparison_frames = []
    for orig, over in zip(original_crop_frames, overlay_frames):
        # 添加标签
        orig_labeled = orig.copy()
        over_labeled = over.copy()
        cv2.putText(orig_labeled, "Original", (5, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(over_labeled, f"Grad-CAM (Class{target})", (5, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        row = np.concatenate([orig_labeled, over_labeled], axis=1)
        comparison_frames.append(row)

    # --- 9. 保存视频 ---
    os.makedirs(output_dir, exist_ok=True)
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    # 热力图叠加视频
    overlay_path = os.path.join(output_dir, f"{video_name}_gradcam.mp4")
    h_out, w_out = overlay_frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(overlay_path, fourcc, 8, (w_out, h_out))
    for frame in overlay_frames:
        writer.write(frame)
    writer.release()

    # 对比视频 (原始 | 热力图)
    compare_path = os.path.join(output_dir, f"{video_name}_gradcam_compare.mp4")
    h_c, w_c = comparison_frames[0].shape[:2]
    writer_c = cv2.VideoWriter(compare_path, fourcc, 8, (w_c, h_c))
    for frame in comparison_frames:
        writer_c.write(frame)
    writer_c.release()

    # --- 10. 保存汇总图 (8帧拼接: 每个temporal token取1帧) ---
    summary_path = os.path.join(output_dir, f"{video_name}_gradcam_summary.png")
    selected_indices = list(range(0, len(overlay_frames), 2))[:T_tokens]  # 每个token取1帧
    if not selected_indices:
        selected_indices = [0]
    selected_overlays = [overlay_frames[i] for i in selected_indices]
    selected_originals = [original_crop_frames[i] for i in selected_indices]

    # 上排: 原始帧, 下排: 热力图
    top_row = np.concatenate(selected_originals, axis=1)
    bot_row = np.concatenate(selected_overlays, axis=1)
    summary = np.concatenate([top_row, bot_row], axis=0)

    # 添加总标题
    header = np.zeros((30, summary.shape[1], 3), dtype=np.uint8)
    cv2.putText(header, f"Pred: Class{pred_class} | {prob_text}", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    summary_with_header = np.concatenate([header, summary], axis=0)
    cv2.imwrite(summary_path, summary_with_header)

    # --- 11. 打印结果 ---
    print(f"\n=> 预测结果:")
    for c in range(num_classes):
        marker = " <--" if c == pred_class else ""
        print(f"  Class {c}: {probs[0, c]:.4f}{marker}")
    print(f"  Grad-CAM 目标类别: Class {target}")
    print(f"\n=> 热力图视频: {overlay_path}")
    print(f"=> 对比视频:   {compare_path}")
    print(f"=> 汇总图:     {summary_path}")


if __name__ == "__main__":
    visualize_gradcam("./assets/CLASS1_0.mp4")
