import os
import tempfile

import cv2
import imageio.v3 as iio
import numpy as np
import timm
import torch
import torch.nn.functional as F
from decord import VideoReader, cpu
from PIL import Image
from torchvision import transforms

from buildmodel import build_model
from utils import make_transforms


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def load_video_frames(video_path, frames_per_clip=16, frame_step=2):
    """Load and sample frames from a video using decord."""
    vr = VideoReader(video_path, num_threads=-1, ctx=cpu(0))
    total_frames = len(vr)

    if total_frames < frames_per_clip * frame_step:
        indices = np.linspace(0, total_frames - 1, num=frames_per_clip).astype(np.int64)
    else:
        clip_len = frames_per_clip * frame_step
        start_idx = (total_frames - clip_len) // 2
        indices = np.arange(start_idx, start_idx + clip_len, frame_step)[:frames_per_clip]

    buffer = vr.get_batch(indices).asnumpy()
    return buffer, indices


def sample_16_frames(video_path):
    """Sample 16 frames evenly spaced across the full video duration.

    Returns (buffer, indices, total_frames).
    """
    vr = VideoReader(video_path, num_threads=-1, ctx=cpu(0))
    total = len(vr)
    if total <= 16:
        indices = np.arange(total, dtype=np.int64)
    else:
        indices = np.linspace(0, total - 1, 16, dtype=np.int64)
    buffer = vr.get_batch(indices).asnumpy()
    return buffer, indices, total


def draw_diagonal_hatch(patch_img, gap=6, color=(180, 50, 50)):
    """Draw diagonal hatching lines onto patch_img in-place (BGR)."""
    h, w = patch_img.shape[:2]
    for offset in range(-h, w, gap):
        pt1 = (offset, 0)
        pt2 = (offset + h, h)
        cv2.line(patch_img, pt1, pt2, color, 1, cv2.LINE_AA)


def frames_to_clip_video(video_path, start_frame_idx, num_frames=10, fps=8):
    """Extract num_frames from video at start_frame_idx, write to temp .mp4."""
    vr = VideoReader(video_path, num_threads=-1, ctx=cpu(0))
    total = len(vr)
    end = min(start_frame_idx + num_frames, total)
    indices = np.arange(start_frame_idx, end, dtype=np.int64)
    if len(indices) < num_frames:
        pad_count = num_frames - len(indices)
        indices = np.concatenate([indices, np.full(pad_count, indices[-1], dtype=np.int64)])
    buffer = vr.get_batch(indices).asnumpy()
    frames_rgb = [cv2.cvtColor(buffer[i], cv2.COLOR_BGR2RGB) for i in range(len(buffer))]
    return frames_to_video_file(frames_rgb, fps=fps, prefix="clip_crop")


def build_16frame_montage_with_timeline(video_path, frame_height=120):
    """Build 16-frame tiled montage + real frame-axis timeline with arrows.

    Top row: 16 evenly-sampled frames tiled horizontally.
    Middle: arrow connectors from each tile down to its position on the real axis.
    Bottom: timeline of ALL original frames. Sampled positions get diagonal hatch
            if prob > 0.5. Clip boxes cover 10 consecutive original frames.

    Returns dict with keys: montage_image, frame_info, clip_regions, clip_video_paths.
    """
    buffer, orig_indices, total_frames = sample_16_frames(video_path)
    orig_h, orig_w = buffer.shape[1], buffer.shape[2]
    aspect = orig_w / max(orig_h, 1)
    frame_w = int(frame_height * aspect)

    # --- Per-frame random data ---
    frame_info = []
    for _ in range(16):
        possibility = round(random.uniform(0.15, 0.85), 3)
        has_lesion = possibility > 0.5
        if has_lesion:
            r = random.random()
            if r < 0.75:
                num_boxes = 1
            elif r < 0.92:
                num_boxes = 2
            else:
                num_boxes = 3
        else:
            num_boxes = 1 if random.random() < 0.10 else 0
        boxes = []
        for _ in range(num_boxes):
            x1 = random.uniform(0.05, 0.35)
            y1 = random.uniform(0.05, 0.35)
            x2 = random.uniform(x1 + 0.15, min(x1 + 0.55, 0.95))
            y2 = random.uniform(y1 + 0.15, min(y1 + 0.55, 0.95))
            boxes.append([x1, y1, x2, y2])
        frame_info.append({
            "boxes_found": num_boxes > 0,
            "num_boxes": num_boxes,
            "boxes": boxes,
            "possibility": possibility,
        })

    # --- Build 16-frame tile row ---
    resized_frames = []
    for i in range(16):
        frame = buffer[i].copy()
        frame_rs = cv2.resize(frame, (frame_w, frame_height))
        for box in frame_info[i]["boxes"]:
            x1, y1, x2, y2 = box
            pt1 = (int(x1 * frame_w), int(y1 * frame_height))
            pt2 = (int(x2 * frame_w), int(y2 * frame_height))
            cv2.rectangle(frame_rs, pt1, pt2, (0, 255, 0), 2)
            cv2.putText(frame_rs, "lesion", (pt1[0], pt1[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        resized_frames.append(frame_rs)

    tile_row_w = 16 * frame_w
    montage = np.concatenate(resized_frames, axis=1)  # (frame_height, tile_row_w, 3)

    # --- Real frame-axis timeline ---
    tl_h = 44
    # Each original frame gets at least 2 px; timeline width may differ from tile row
    px_per_frame = max(3, tile_row_w // total_frames)
    tl_w = px_per_frame * total_frames

    timeline = np.zeros((tl_h, tl_w, 3), dtype=np.uint8)
    timeline[:] = (220, 210, 215)  # light red BGR

    # Draw sampled-position markers
    for i in range(16):
        orig_idx = int(orig_indices[i])
        x_start = orig_idx * px_per_frame
        x_end = x_start + px_per_frame
        if frame_info[i]["possibility"] > 0.5:
            draw_diagonal_hatch(timeline[:, x_start:x_end, :])
        # Frame number label
        cv2.putText(timeline, str(i + 1), (x_start + 1, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, (60, 60, 60), 1)

    # White tick marks at each sampled position
    for i in range(16):
        x = int(orig_indices[i]) * px_per_frame + px_per_frame // 2
        cv2.line(timeline, (x, tl_h - 10), (x, tl_h - 1), (255, 255, 255), 1)

    # --- Arrow connector row ---
    # Unify widths: max of tile row and timeline
    canvas_w = max(tile_row_w, tl_w)
    arrow_h = 22

    # Pad montage if narrower
    if tile_row_w < canvas_w:
        pad = np.ones((frame_height, canvas_w - tile_row_w, 3), dtype=np.uint8) * 240
        montage = np.concatenate([montage, pad], axis=1)
    # Pad timeline if narrower
    if tl_w < canvas_w:
        pad = np.ones((tl_h, canvas_w - tl_w, 3), dtype=np.uint8) * 240
        timeline = np.concatenate([timeline, pad], axis=1)

    arrow_row = np.ones((arrow_h, canvas_w, 3), dtype=np.uint8) * 240

    for i in range(16):
        tile_cx = i * frame_w + frame_w // 2
        tl_cx = int(orig_indices[i]) * px_per_frame + px_per_frame // 2
        # Vertical line from tile down, then angled to timeline position
        mid_y = arrow_h // 2
        cv2.line(arrow_row, (tile_cx, 0), (tile_cx, mid_y), (130, 130, 130), 1, cv2.LINE_AA)
        cv2.line(arrow_row, (tile_cx, mid_y), (tl_cx, arrow_h - 2), (130, 130, 130), 1, cv2.LINE_AA)
        # Small arrowhead
        cv2.circle(arrow_row, (tl_cx, arrow_h - 2), 2, (130, 130, 130), -1, cv2.LINE_AA)

    # --- Clip detection ---
    clips = []
    i = 0
    while i < 16:
        if frame_info[i]["possibility"] > 0.5:
            j = i
            while j < 16 and frame_info[j]["possibility"] > 0.5:
                j += 1
            if j - i >= 3 and len(clips) < 5:
                clips.append({
                    "start_frame": i,
                    "end_frame": j - 1,
                    "label": f"clip{len(clips) + 1}",
                    "start_orig_idx": int(orig_indices[i]),
                })
            i = j
        else:
            i += 1

    # --- Crop clips: 10 consecutive original frames ---
    clip_video_paths = []
    for clip in clips:
        clip_path = frames_to_clip_video(video_path, clip["start_orig_idx"], num_frames=10)
        clip_video_paths.append(clip_path)

    # --- Annotate timeline with clip regions (10 original-frame cells) ---
    for clip in clips:
        x1 = clip["start_orig_idx"] * px_per_frame
        x2 = min((clip["start_orig_idx"] + 10) * px_per_frame, tl_w)
        cv2.rectangle(timeline, (x1, 0), (x2, tl_h - 1), (0, 210, 0), 2)
        cv2.putText(timeline, clip["label"], (x1 + 3, tl_h - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 180, 0), 1)

    # --- Stack vertically: montage + arrows + timeline ---
    combined = np.concatenate([montage, arrow_row, timeline], axis=0)

    # --- BGR → RGB ---
    montage_rgb = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)

    return {
        "montage_image": montage_rgb,
        "frame_info": frame_info,
        "clip_regions": clips,
        "clip_video_paths": clip_video_paths,
    }


def prepare_model_input(buffer, transform, device):
    """Apply eval transforms and format for VJEPA model input."""
    clips = transform(buffer)
    return [[c.unsqueeze(0).to(device) for c in clips]]


def frames_to_video_file(frames, fps=8, prefix="gradcam"):
    """Write RGB numpy frames to a temp mp4 file (H.264). Returns file path."""
    tmp_dir = tempfile.mkdtemp(prefix="gradio_")
    path = os.path.join(tmp_dir, f"{prefix}.mp4")
    iio.imwrite(path, frames, fps=fps, codec="libx264")
    return path


YOLO_MODEL_PATH = "./QC/best_640_s_60e(2).pt"
SVM_MODEL_PATH = "./QC/mobilenetv3_small_075_yl_241222(3).pth"
YOLO_IMG_SIZE = 256
YOLO_EXCLUDE_CLASS = 1
YOLO_CONF_THRESHOLD = 0.5
BATCH_SIZE = 16


def resolve_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_mobilenet(device):
    model = timm.create_model("mobilenetv3_small_075", pretrained=False, in_chans=3, num_classes=2).to(device)
    if not os.path.exists(SVM_MODEL_PATH):
        raise FileNotFoundError(f"MobileNet checkpoint not found: {SVM_MODEL_PATH}")
    model.load_state_dict(torch.load(SVM_MODEL_PATH, map_location=device))
    model.eval()
    return model


def build_yolo(device):
    from ultralytics import YOLO

    model = YOLO(YOLO_MODEL_PATH)
    model(np.zeros((YOLO_IMG_SIZE, YOLO_IMG_SIZE, 3), dtype=np.uint8), device=str(device), verbose=False)
    return model


def load_stage1_models(device):
    mobilenet_model = build_mobilenet(device)
    try:
        yolo_model = build_yolo(device)
        yolo_device = device
    except torch.AcceleratorError as e:
        if device.type != "cuda":
            raise
        print(f"YOLO CUDA init failed, fallback to CPU: {e}")
        yolo_model = build_yolo(torch.device("cpu"))
        yolo_device = torch.device("cpu")
    return mobilenet_model, yolo_model, yolo_device


def build_stage1_preprocess():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])


def draw_mobilenet_score(frame, score):
    valid = 1.0 - score
    text = f"Valid {valid:.2f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.42
    text_thickness = 1
    (tw, th), baseline = cv2.getTextSize(text, font, scale, text_thickness)
    x = frame.shape[1] - tw - 4
    y = th + 4
    cv2.putText(
        frame,
        text,
        (x, y),
        font,
        scale,
        (245, 245, 245),
        text_thickness,
        cv2.LINE_AA,
    )


def draw_yolo_boxes(frame, boxes):
    color = (0, 215, 120)
    for box in boxes:
        x1, y1, x2, y2 = map(int, box["xyxy"])
        conf = float(box["conf"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
        label = f"{conf:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.42
        text_thickness = 1
        (_, th), baseline = cv2.getTextSize(label, font, scale, text_thickness)
        text_y = y1 - 2
        if text_y - th < 0:
            text_y = min(frame.shape[0] - baseline - 1, y1 + th + 2)
        cv2.putText(
            frame,
            label,
            (x1 + 1, text_y),
            font,
            scale,
            (245, 245, 245),
            text_thickness,
            cv2.LINE_AA,
        )


def infer_stage1_batch(batch_frames, start_idx, mobilenet_model, yolo_model, preprocess, mobilenet_device, yolo_device):
    imgs_pil = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in batch_frames]
    mobilenet_tensors = torch.stack([preprocess(img) for img in imgs_pil]).to(mobilenet_device)

    with torch.inference_mode():
        logits = mobilenet_model(mobilenet_tensors)
        probs = F.softmax(logits, dim=1)
        mobilenet_scores = probs[:, 1].cpu().tolist()

    yolo_scores = [0.0] * len(batch_frames)
    yolo_boxes = [[] for _ in batch_frames]
    yolo_indices = [idx for idx, score in enumerate(mobilenet_scores) if score < 0.5]
    if yolo_indices:
        yolo_inputs = [batch_frames[idx] for idx in yolo_indices]
        yolo_results = yolo_model(
            yolo_inputs,
            imgsz=YOLO_IMG_SIZE,
            conf=YOLO_CONF_THRESHOLD,
            device=str(yolo_device),
            verbose=False,
        )
        for result_idx, batch_idx in enumerate(yolo_indices):
            result = yolo_results[result_idx]
            if result.boxes is None:
                continue
            frame_boxes = []
            for box in result.boxes:
                if int(box.cls[0]) == YOLO_EXCLUDE_CLASS:
                    continue
                conf = float(box.conf[0])
                xyxy = [float(v) for v in box.xyxy[0].tolist()]
                frame_boxes.append({
                    "xyxy": xyxy,
                    "conf": conf,
                    "cls": int(box.cls[0]),
                })
            yolo_boxes[batch_idx] = frame_boxes
            yolo_scores[batch_idx] = max((box["conf"] for box in frame_boxes), default=0.0)

    batch_results = []
    annotated_frames = []
    for offset, frame in enumerate(batch_frames):
        mobilenet_score = mobilenet_scores[offset]
        yolo_score = yolo_scores[offset]
        boxes = yolo_boxes[offset]

        annotated = frame.copy()
        draw_yolo_boxes(annotated, boxes)
        draw_mobilenet_score(annotated, mobilenet_score)
        annotated_frames.append(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))

        batch_results.append({
            "frame_idx": start_idx + offset,
            "yolo_score": round(yolo_score, 4),
            "mobilenet_score": round(mobilenet_score, 4),
            "yolo_boxes": [
                {
                    "xyxy": [round(v, 2) for v in box["xyxy"]],
                    "conf": round(box["conf"], 4),
                    "cls": box["cls"],
                }
                for box in boxes
            ],
        })
    return batch_results, annotated_frames


# ---------------------------------------------------------------------------
# Stage 1: Rapid Negative Filtering (real)
# ---------------------------------------------------------------------------

class Stage1Filter:
    """MobileNet + YOLO stage-1 screening."""

    def __init__(self, device=None, batch_size=BATCH_SIZE):
        self.device = device or resolve_device()
        self.batch_size = batch_size
        self.mobilenet_model, self.yolo_model, self.yolo_device = load_stage1_models(self.device)
        self.preprocess = build_stage1_preprocess()

    def run(self, video_path):
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_results = []
        annotated_frames = []
        batch_frames = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            batch_frames.append(frame)
            if len(batch_frames) == self.batch_size:
                batch_results, batch_annotated_frames = infer_stage1_batch(
                    batch_frames,
                    frame_idx - len(batch_frames) + 1,
                    self.mobilenet_model,
                    self.yolo_model,
                    self.preprocess,
                    self.device,
                    self.yolo_device,
                )
                frame_results.extend(batch_results)
                annotated_frames.extend(batch_annotated_frames)
                batch_frames = []
            frame_idx += 1

        if batch_frames:
            batch_results, batch_annotated_frames = infer_stage1_batch(
                batch_frames,
                frame_idx - len(batch_frames),
                self.mobilenet_model,
                self.yolo_model,
                self.preprocess,
                self.device,
                self.yolo_device,
            )
            frame_results.extend(batch_results)
            annotated_frames.extend(batch_annotated_frames)

        cap.release()

        if not frame_results:
            raise RuntimeError(f"No frames decoded from video: {video_path}")

        annotated_video_path = frames_to_video_file(annotated_frames, fps=fps, prefix="stage1_annotated")
        valid_frames = sum(1 for result in frame_results if result["mobilenet_score"] < 0.5)
        detected_frames = sum(1 for result in frame_results if result["yolo_boxes"])
        max_yolo_conf = max((result["yolo_score"] for result in frame_results), default=0.0)
        mean_valid_score = sum(1.0 - result["mobilenet_score"] for result in frame_results) / len(frame_results)
        has_lesions = valid_frames > 0 and detected_frames > 0

        return {
            "status": "pass" if has_lesions else "stop",
            "message": (
                f"{valid_frames}/{len(frame_results)} valid frames, "
                f"{detected_frames} frames with detections, max YOLO conf {max_yolo_conf:.3f}."
            ),
            "classification": {
                "has_lesions": has_lesions,
                "confidence": float(mean_valid_score),
                "model_name": "MobileNet-V3 + YOLO",
                "valid_frames": valid_frames,
                "total_frames": len(frame_results),
            },
            "detection": {
                "boxes_found": detected_frames > 0,
                "num_boxes": sum(len(result["yolo_boxes"]) for result in frame_results),
                "frames_with_boxes": detected_frames,
                "max_confidence": float(max_yolo_conf),
                "model_name": "YOLO",
            },
            "frame_results": frame_results,
            "annotated_video_path": annotated_video_path,
        }


# ---------------------------------------------------------------------------
# Stage 3: VJEPA2 3-class classification (real)
# ---------------------------------------------------------------------------

class Stage3Classifier:
    """Real VJEPA2 ViT-Giant classification with Grad-CAM."""

    def __init__(self, checkpoint_path, device=None,
                 img_size=224, frames_per_clip=16,
                 num_classes=3, num_heads=16, num_probe_blocks=1):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.img_size = img_size
        self.frames_per_clip = frames_per_clip
        self.num_classes = num_classes
        self.frame_step = 2
        self.tubelet_size = 2

        self.model, self.classifier = build_model(
            checkpoint_path=checkpoint_path,
            resolution=img_size,
            frames_per_clip=frames_per_clip,
            num_classes=num_classes,
            num_heads=num_heads,
            num_probe_blocks=num_probe_blocks,
        )
        self.model = self.model.to(self.device).eval()
        self.classifier = self.classifier.to(self.device).eval()
        self.transform = make_transforms(
            training=False, num_views_per_clip=1, crop_size=img_size,
        )

    def predict(self, video_path):
        """Run VJEPA2 classification with mixed precision."""
        buffer, _ = load_video_frames(video_path, self.frames_per_clip, self.frame_step)
        clips_input = prepare_model_input(buffer, self.transform, self.device)

        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.float16):
                features = self.model(clips_input)
                logits_list = [self.classifier(f) for f in features]
                probs = sum(F.softmax(l, dim=1) for l in logits_list) / len(logits_list)

        probs_np = probs[0].cpu().float().numpy()
        pred_class = int(probs_np.argmax())

        return {
            "probs": {f"class_{c}": float(probs_np[c]) for c in range(self.num_classes)},
            "predicted_class": pred_class,
            "confidence": float(probs_np[pred_class]),
        }

    def gradcam(self, video_path, target_class=None, alpha=0.2):
        """Run Grad-CAM visualization. Returns in-memory arrays (RGB)."""
        buffer, indices = load_video_frames(video_path, self.frames_per_clip, self.frame_step)
        clips_input = prepare_model_input(buffer, self.transform, self.device)

        # Feature extraction (no grad)
        with torch.no_grad():
            features = self.model(clips_input)

        feat = features[0].float().detach().requires_grad_(True)

        # Classifier forward + backward
        logits = self.classifier(feat)
        pred_class = logits.argmax(dim=1).item()
        target = target_class if target_class is not None else pred_class

        self.classifier.zero_grad()
        logits[0, target].backward()

        # Grad-CAM computation
        activations = feat.detach()
        gradients = feat.grad.detach()
        weights = gradients.mean(dim=1)
        cam = (weights.unsqueeze(1) * activations).sum(dim=-1)
        cam = F.relu(cam).squeeze(0).cpu().numpy()

        T_tokens = self.frames_per_clip // self.tubelet_size
        H_patches = self.img_size // 16
        W_patches = self.img_size // 16
        cam = cam.reshape(T_tokens, H_patches, W_patches)

        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 0:
            cam = (cam - cam_min) / (cam_max - cam_min)

        # Overlay on original frames
        original_frames = buffer
        short_side = int(self.img_size * 256 / 224)
        probs = F.softmax(logits.detach(), dim=1)

        overlay_frames = []
        original_crop_frames = []

        for frame_idx in range(len(indices)):
            frame = original_frames[frame_idx].copy()
            h, w = frame.shape[:2]

            if h < w:
                new_h, new_w = short_side, int(w * short_side / h)
            else:
                new_h, new_w = int(h * short_side / w), short_side
            frame_resized = cv2.resize(frame, (new_w, new_h))

            start_h = (new_h - self.img_size) // 2
            start_w = (new_w - self.img_size) // 2
            frame_crop = frame_resized[start_h:start_h + self.img_size,
                                       start_w:start_w + self.img_size]
            original_crop_frames.append(frame_crop.copy())

            token_idx = min(frame_idx // self.tubelet_size, T_tokens - 1)
            heatmap = cam[token_idx]

            heatmap_up = cv2.resize(heatmap, (self.img_size, self.img_size),
                                    interpolation=cv2.INTER_CUBIC)
            heatmap_uint8 = np.uint8(255 * heatmap_up)
            heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(frame_crop, 1 - alpha, heatmap_color, alpha, 0)
            overlay_frames.append(overlay)

        # Comparison frames (original | heatmap) side by side
        comparison_frames = []
        for orig, over in zip(original_crop_frames, overlay_frames):
            orig_l = orig.copy()
            over_l = over.copy()
            cv2.putText(orig_l, "Original", (5, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(over_l, f"Grad-CAM (Class {target})", (5, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            comparison_frames.append(np.concatenate([orig_l, over_l], axis=1))

        # Summary image (top row: original, bottom row: heatmap)
        selected_indices = list(range(0, len(overlay_frames), 2))[:T_tokens]
        if not selected_indices:
            selected_indices = [0]
        top_row = np.concatenate([original_crop_frames[i] for i in selected_indices], axis=1)
        bot_row = np.concatenate([overlay_frames[i] for i in selected_indices], axis=1)
        summary = np.concatenate([top_row, bot_row], axis=0)

        header = np.zeros((30, summary.shape[1], 3), dtype=np.uint8)
        prob_text = "  ".join([f"C{c}:{probs[0, c]:.3f}" for c in range(self.num_classes)])
        pred_label = "Class 0 (Low Risk / No Lesion)" if pred_class == 0 else f"Class {pred_class}"
        cv2.putText(header, f"Pred: {pred_label} | {prob_text}", (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        summary_with_header = np.concatenate([header, summary], axis=0)

        # Convert BGR → RGB for Gradio
        overlay_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in overlay_frames]
        compare_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in comparison_frames]
        summary_rgb = cv2.cvtColor(summary_with_header, cv2.COLOR_BGR2RGB)

        return {
            "overlay_frames": overlay_rgb,
            "comparison_frames": compare_rgb,
            "summary_image": summary_rgb,
            "pred_class": pred_class,
            "probs": {f"class_{c}": float(probs[0, c]) for c in range(self.num_classes)},
        }


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------

class PipelineRunner:
    """Orchestrates the 3-stage pipeline."""

    def __init__(self, stage3_checkpoint_path):
        self.stage1 = Stage1Filter()
        self.stage3 = None
        self.stage3_checkpoint = stage3_checkpoint_path

    def _ensure_stage3(self):
        if self.stage3 is None:
            self.stage3 = Stage3Classifier(self.stage3_checkpoint)

    def run_stage1(self, video_path):
        stage1_result = self.stage1.run(video_path)
        if stage1_result["status"] == "stop":
            return {
                "status": "stop",
                "classification": stage1_result["classification"],
                "detection": stage1_result["detection"],
                "message": stage1_result["message"],
                "frame_results": stage1_result["frame_results"],
                "annotated_video_path": stage1_result["annotated_video_path"],
            }

        return {
            "status": "pass",
            "classification": stage1_result["classification"],
            "detection": stage1_result["detection"],
            "message": stage1_result["message"],
            "frame_results": stage1_result["frame_results"],
            "annotated_video_path": stage1_result["annotated_video_path"],
        }

    def run_stage3(self, video_path, clip_paths=None):
        self._ensure_stage3()

        pred = self.stage3.predict(video_path)
        gradcam_result = self.stage3.gradcam(video_path, target_class=pred["predicted_class"])
        compare_path = frames_to_video_file(gradcam_result["comparison_frames"], prefix="compare")
        return {
            "status": "complete",
            "prediction": pred,
            "clip_results": None,
            "top_clip_index": -1,
            "gradcam_compare_path": compare_path,
            "gradcam_summary": gradcam_result["summary_image"],
        }
