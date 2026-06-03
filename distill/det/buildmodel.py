"""
Model builders for detection distillation.

Teacher: ViT-G (via root buildmodel.py)
Student: YOLO11m backbone (via ultralytics, hook-based feature extraction)
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

_VJEPA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _VJEPA_ROOT not in sys.path:
    sys.path.insert(0, _VJEPA_ROOT)


# ---------------------------------------------------------------------------
# Feature hook
# ---------------------------------------------------------------------------

class FeatureHook:
    """Captures intermediate layer output from a forward hook."""

    def __init__(self):
        self.features = None

    def hook_fn(self, module, input, output):
        self.features = output


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class YOLOFeatureAdapter(nn.Module):
    """1×1 Conv projecting YOLO backbone channels → teacher embed_dim (1408)."""

    def __init__(self, in_channels, out_channels=1408):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


# ---------------------------------------------------------------------------
# Teacher loading & feature extraction (2D images)
# ---------------------------------------------------------------------------

def load_teacher(checkpoint_path, num_frames=2, resolution=224, device='cuda'):
    """
    Load ViT-G teacher encoder from checkpoint.

    Uses root buildmodel.build_model() — same as train.py.
    num_frames=2 is the minimum for tubelet_size=2.
    """
    from buildmodel import build_model

    encoder, _ = build_model(
        checkpoint_path=checkpoint_path,
        resolution=resolution,
        frames_per_clip=num_frames,
        num_classes=3,
        num_heads=16,
        num_probe_blocks=1,
        model_name='vit_giant_xformers',
    )
    encoder = encoder.to(device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    return encoder


def extract_teacher_2d_features(teacher_encoder, images_2d, num_frames=2,
                                 teacher_res=224):
    """
    Extract spatial features from ViT-G for 2D detection images.

    Duplicates the 2D image to num_frames along the temporal axis (minimum 2
    for tubelet_size=2), producing a spatial feature map.

    Args:
        teacher_encoder: ClipAggregation(ViT) from load_teacher()
        images_2d:      [B, C, H, W]  — single-frame YOLO input (0-1 range)
        num_frames:      temporal copies (default 2, tubelet min)
        teacher_res:     spatial resolution expected by teacher (default 224)

    Returns:
        [B, 1408, h_patches, w_patches]  spatial feature map
    """
    B, C, H, W = images_2d.shape

    # YOLO images are in [0, 1] range; ViT-G expects ImageNet normalization
    imagenet_mean = torch.tensor([0.485, 0.456, 0.406],
                                 device=images_2d.device).view(1, 3, 1, 1)
    imagenet_std = torch.tensor([0.229, 0.224, 0.225],
                                device=images_2d.device).view(1, 3, 1, 1)
    images_2d = (images_2d - imagenet_mean) / imagenet_std

    # Resize to teacher resolution if needed
    if H != teacher_res or W != teacher_res:
        images_2d = F.interpolate(
            images_2d, size=(teacher_res, teacher_res),
            mode='bilinear', align_corners=False,
        )

    # Build pseudo-video
    video_3d = images_2d.unsqueeze(2).repeat(1, 1, num_frames, 1, 1)  # [B, C, T, H, W]

    clips = [[video_3d]]

    with torch.no_grad():
        outputs = teacher_encoder(clips)       # list of [B, N, 1408]

    feat = outputs[0]                           # [B, N, 1408]

    # Reshape to spatial feature map
    # N = (T/tubelet_size) * (H/patch_size) * (W/patch_size)
    T_tokens = num_frames // 2                  # tubelet_size=2 → 1
    S_tokens = feat.size(1) // T_tokens          # total spatial tokens

    # Assume square spatial grid
    h_patches = int(S_tokens ** 0.5)
    w_patches = h_patches

    feat = feat.view(B, T_tokens, h_patches, w_patches, -1)  # [B, 1, Hp, Wp, 1408]
    feat = feat.squeeze(1).permute(0, 3, 1, 2)                # [B, 1408, Hp, Wp]
    return feat


# ---------------------------------------------------------------------------
# YOLO backbone helpers
# ---------------------------------------------------------------------------

def get_yolo_backbone_channels(yolo_model, hook_layer, imgsz=256):
    """Auto-detect output channels of the hooked backbone layer."""
    device = next(yolo_model.model.parameters()).device
    dummy = torch.randn(1, 3, imgsz, imgsz).to(device)
    x = dummy
    for i in range(hook_layer + 1):
        x = yolo_model.model[i](x)
    return x.size(1)


def load_yolo_distilled_backbone(yolo_model, weight_path):
    """
    Load distilled YOLO backbone weights from a Stage-1 KD checkpoint.

    Supports two checkpoint formats:
      - Lightweight: dict with 'backbone_state_dict' key (from save_model in train.py)
      - Full: dict with 'model' key (standard ultralytics best.pt / last.pt)

    Args:
        yolo_model: ultralytics YOLO model (DetectionModel)
        weight_path: path to distilled_backbone.pt or full checkpoint

    Returns:
        yolo_model with backbone weights loaded
    """
    ckpt = torch.load(weight_path, map_location='cpu', weights_only=False)

    if 'backbone_state_dict' in ckpt:
        backbone_state = ckpt['backbone_state_dict']
    elif 'model' in ckpt:
        # Full checkpoint — extract backbone layers only.
        # Skip the Detect head (class-dependent channel sizes) and any layer
        # whose shape doesn't match (class count differs between datasets).
        full_state = ckpt['model'].state_dict()
        model_state = yolo_model.model.state_dict()
        backbone_state = {}
        skipped = 0
        for k, v in full_state.items():
            if k not in model_state:
                continue
            if v.shape != model_state[k].shape:
                skipped += 1
                continue
            backbone_state[k] = v
        rank = int(os.environ.get('LOCAL_RANK', 0))
        if rank == 0 and skipped > 0:
            print(f'skipped {skipped} incompatible head layers (class count mismatch)')
    else:
        raise KeyError(
            f'Checkpoint must contain "backbone_state_dict" or "model" key. '
            f'Found: {list(ckpt.keys())}'
        )

    model_state = yolo_model.model.state_dict()

    model_keys = set(model_state.keys())
    ckpt_keys = set(backbone_state.keys())
    matched = len(model_keys & ckpt_keys)
    extra = len(ckpt_keys - model_keys)
    missing = len(model_keys - ckpt_keys)

    filtered = {k: v for k, v in backbone_state.items() if k in model_state}
    yolo_model.model.load_state_dict(filtered, strict=False)

    rank = int(os.environ.get('LOCAL_RANK', 0))
    if rank == 0:
        print(f'loaded pretrained model: matched {matched}, extra {extra}, missing {missing}')
    return yolo_model
