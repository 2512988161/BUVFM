import os
import random
import tempfile

import cv2
import imageio.v3 as iio
import numpy as np
import torch
import torch.nn.functional as F
from decord import VideoReader, cpu

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


# ---------------------------------------------------------------------------
# Stage 1: Rapid Negative Filtering (placeholder)
# ---------------------------------------------------------------------------

class Stage1Filter:
    """Lightweight classifier + detector placeholder.

    Replace classify() and detect_boxes() internals with real
    MobileNet / YOLO when available.
    """

    def __init__(self, device=None):
        self.device = device

    def classify(self, video_path):
        """Check if video likely contains lesions. 70 % positive rate."""
        has_lesions = random.random() < 0.70
        return {
            "has_lesions": has_lesions,
            "confidence": round(random.uniform(0.55, 0.95), 3),
            "model_name": "MobileNet-V3 (placeholder)",
        }

    def detect_boxes(self, video_path):
        """Run lightweight detector for bounding boxes. 80 % box-found rate."""
        boxes_found = random.random() < 0.80
        num_boxes = random.randint(1, 3) if boxes_found else 0
        boxes = []
        for _ in range(num_boxes):
            x1 = random.uniform(0.05, 0.35)
            y1 = random.uniform(0.05, 0.35)
            x2 = random.uniform(x1 + 0.15, min(x1 + 0.55, 0.95))
            y2 = random.uniform(y1 + 0.15, min(y1 + 0.55, 0.95))
            boxes.append([x1, y1, x2, y2])

        annotated_frame = None
        if boxes_found:
            try:
                buffer, _ = load_video_frames(video_path)
                mid_frame = buffer[len(buffer) // 2].copy()
                h, w = mid_frame.shape[:2]
                for box in boxes:
                    pt1 = (int(box[0] * w), int(box[1] * h))
                    pt2 = (int(box[2] * w), int(box[3] * h))
                    cv2.rectangle(mid_frame, pt1, pt2, (0, 255, 0), 2)
                    cv2.putText(mid_frame, "lesion", (pt1[0], pt1[1] - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                annotated_frame = cv2.cvtColor(mid_frame, cv2.COLOR_BGR2RGB)
            except Exception:
                pass

        return {
            "boxes_found": boxes_found,
            "num_boxes": num_boxes,
            "boxes": boxes,
            "annotated_frame": annotated_frame,
            "model_name": "YOLOv8-nano (placeholder)",
        }


# ---------------------------------------------------------------------------
# Stage 2: Medium-weight refinement (placeholder)
# ---------------------------------------------------------------------------

class Stage2Filter:
    """nmODE-ResNet placeholder. Replace with real model when available."""

    def __init__(self, device=None):
        self.device = device

    def classify(self, video_path):
        """Further filter non-lesion videos. 80 % positive rate."""
        has_lesions = random.random() < 0.80
        return {
            "has_lesions": has_lesions,
            "confidence": round(random.uniform(0.60, 0.98), 3),
            "model_name": "nmODE-ResNet18 (placeholder)",
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
        cv2.putText(header, f"Pred: Class {pred_class} | {prob_text}", (10, 22),
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
        self.stage2 = Stage2Filter()
        self.stage3 = None
        self.stage3_checkpoint = stage3_checkpoint_path

    def _ensure_stage3(self):
        if self.stage3 is None:
            self.stage3 = Stage3Classifier(self.stage3_checkpoint)

    def run_stage1(self, video_path):
        cls_result = self.stage1.classify(video_path)
        if not cls_result["has_lesions"]:
            return {
                "status": "stop",
                "classification": cls_result,
                "detection": None,
                "message": "No lesions detected by classifier. Pipeline stopped.",
                "montage_image": None,
                "frame_info": None,
                "clip_regions": [],
                "clip_video_paths": [],
            }

        montage_result = build_16frame_montage_with_timeline(video_path)

        any_boxes = any(fi["boxes_found"] for fi in montage_result["frame_info"])
        if not any_boxes:
            return {
                "status": "stop",
                "classification": cls_result,
                "detection": {"boxes_found": False, "num_boxes": 0},
                "message": "No lesion regions found by detector. Pipeline stopped.",
                "montage_image": montage_result["montage_image"],
                "frame_info": montage_result["frame_info"],
                "clip_regions": [],
                "clip_video_paths": [],
            }

        total_boxes = sum(fi["num_boxes"] for fi in montage_result["frame_info"])
        n_frames = sum(1 for fi in montage_result["frame_info"] if fi["boxes_found"])
        return {
            "status": "pass",
            "classification": cls_result,
            "detection": {
                "boxes_found": True,
                "num_boxes": total_boxes,
                "frames_with_boxes": n_frames,
            },
            "message": (f"{total_boxes} lesion region(s) across {n_frames} frames. "
                        f"{len(montage_result['clip_video_paths'])} clip(s) extracted."),
            "montage_image": montage_result["montage_image"],
            "frame_info": montage_result["frame_info"],
            "clip_regions": montage_result["clip_regions"],
            "clip_video_paths": montage_result["clip_video_paths"],
        }

    def run_stage2(self, video_path):
        result = self.stage2.classify(video_path)
        if not result["has_lesions"]:
            return {
                "status": "stop",
                "classification": result,
                "message": "No lesions confirmed. Pipeline stopped.",
            }
        return {
            "status": "pass",
            "classification": result,
            "message": "Lesions confirmed. Proceeding to classification...",
        }

    def run_stage3(self, video_path, clip_paths=None):
        self._ensure_stage3()

        if not clip_paths:
            # Fallback: single video
            pred = self.stage3.predict(video_path)
            gradcam_result = self.stage3.gradcam(video_path, target_class=pred["predicted_class"])
            overlay_path = frames_to_video_file(gradcam_result["overlay_frames"], prefix="overlay")
            compare_path = frames_to_video_file(gradcam_result["comparison_frames"], prefix="compare")
            return {
                "status": "complete",
                "prediction": pred,
                "clip_results": None,
                "top_clip_index": -1,
                "gradcam_compare_path": compare_path,
                "gradcam_summary": gradcam_result["summary_image"],
            }

        # Multi-clip: classify each clip, pick top-risk
        clip_predictions = []
        for i, cp in enumerate(clip_paths):
            pred = self.stage3.predict(cp)
            clip_predictions.append({
                "clip_index": i,
                "prediction": pred,
                "video_path": cp,
            })

        top = max(clip_predictions,
                  key=lambda x: (x["prediction"]["predicted_class"],
                                 x["prediction"]["confidence"]))
        top_idx = top["clip_index"]

        gradcam_result = self.stage3.gradcam(
            clip_paths[top_idx], target_class=top["prediction"]["predicted_class"])
        overlay_path = frames_to_video_file(gradcam_result["overlay_frames"], prefix="overlay")
        compare_path = frames_to_video_file(gradcam_result["comparison_frames"], prefix="compare")

        return {
            "status": "complete",
            "prediction": top["prediction"],
            "clip_results": clip_predictions,
            "top_clip_index": top_idx,
            "gradcam_compare_path": compare_path,
            "gradcam_summary": gradcam_result["summary_image"],
        }
