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
                "annotated_frame": None,
            }
        det_result = self.stage1.detect_boxes(video_path)
        if not det_result["boxes_found"]:
            return {
                "status": "stop",
                "classification": cls_result,
                "detection": det_result,
                "message": "No lesion regions found by detector. Pipeline stopped.",
                "annotated_frame": None,
            }
        return {
            "status": "pass",
            "classification": cls_result,
            "detection": det_result,
            "message": f"{det_result['num_boxes']} lesion region(s) detected. Proceeding...",
            "annotated_frame": det_result["annotated_frame"],
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

    def run_stage3(self, video_path):
        self._ensure_stage3()
        pred = self.stage3.predict(video_path)
        gradcam_result = self.stage3.gradcam(video_path, target_class=pred["predicted_class"])

        overlay_path = frames_to_video_file(gradcam_result["overlay_frames"], prefix="overlay")
        compare_path = frames_to_video_file(gradcam_result["comparison_frames"], prefix="compare")

        return {
            "status": "complete",
            "prediction": pred,
            "gradcam_overlay_path": overlay_path,
            "gradcam_compare_path": compare_path,
            "gradcam_summary": gradcam_result["summary_image"],
        }
