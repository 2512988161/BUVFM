"""
Model builders for classification distillation.

Teacher: ViT-G (via root buildmodel.py)
Student: MobileNetV3-Small (via timm)
"""

import os
import sys
import torch
import torch.nn as nn
import timm

_VJEPA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _VJEPA_ROOT not in sys.path:
    sys.path.insert(0, _VJEPA_ROOT)


# ---------------------------------------------------------------------------
# Student backbone — MobileNetV3-Small split at 14×14
# ---------------------------------------------------------------------------

class MobileNetV3Backbone(nn.Module):
    """
    MobileNetV3-Small split at the last 14×14 feature map (blocks[3]).

    Full pipeline (224×224 input):
      conv_stem → bn1 → act1 → blocks[0](stride2) → blocks[1](stride2)
      → blocks[2](stride2→14×14) → blocks[3](stride1→14×14)   ← backbone
      → blocks[4](stride2→7×7) → blocks[5] → conv_head → pool → fc   ← head

    forward() returns blocks[3] output: [B, 40, 14, 14] (mobilenetv3_small_075).
    forward_full() returns logits [B, num_classes].
    """

    def __init__(self, model_name="mobilenetv3_small_075", num_classes=2,
                 image_size=224):
        super().__init__()
        full = timm.create_model(model_name, pretrained=False, in_chans=3,
                                 num_classes=num_classes)

        # ---- find 14×14 split point inside blocks ----
        dummy = torch.randn(1, 3, image_size, image_size)
        x = full.conv_stem(dummy)
        x = full.bn1(x)
        x = full.act1(x)
        split_idx = 0
        for i, block in enumerate(full.blocks):
            x = block(x)
            if x.shape[-1] == 14:
                split_idx = i + 1
                self._feature_channels = x.shape[1]
        if split_idx == 0:
            raise RuntimeError(f"No 14×14 feature map found for input {image_size}")

        # ---- surgery ----
        self.conv_stem = full.conv_stem
        self.bn1 = full.bn1
        self.act1 = full.act1
        self.blocks_early = full.blocks[:split_idx]   # up to last 14×14 layer
        self.blocks_late = full.blocks[split_idx:]    # 14×14 → 7×7
        self.conv_head = full.conv_head
        self.act2 = full.act2
        self.global_pool = full.global_pool
        self.classifier = full.classifier

        self.model_name = model_name
        self.num_classes = num_classes
        self._feature_size = 14

    def forward(self, x):
        """Return features at 14×14 resolution."""
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.blocks_early(x)
        return x

    def forward_full(self, x):
        """Full forward → logits [B, num_classes]."""
        x = self.forward(x)
        x = self.blocks_late(x)
        x = self.conv_head(x)
        x = self.act2(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    @property
    def feature_channels(self):
        return self._feature_channels

    @property
    def feature_size(self):
        return self._feature_size


# ---------------------------------------------------------------------------
# Projection / temporal alignment
# ---------------------------------------------------------------------------

class Student2TeacherProjector(nn.Module):
    """1×1 conv projecting student channels → teacher embed_dim."""

    def __init__(self, in_channels=40, out_channels=1408):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.proj(x)


class TemporalAggregator(nn.Module):
    """
    Pools T=16 student frames → T'=8 temporal tokens (matching teacher tubelet_size=2).

    AvgPool3d over temporal dimension only (kernel=2, stride=2).
    """

    def __init__(self):
        super().__init__()
        self.pool = nn.AvgPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1))

    def forward(self, x, T=16):
        """
        Args:
            x: [B*T, C, H, W]  projected student features
            T: number of input frames
        Returns:
            [B, T/2, C, H, W]
        """
        B_T, C, H, W = x.shape
        B = B_T // T
        x = x.view(B, T, C, H, W).permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]
        x = self.pool(x)                                      # [B, C, T/2, H, W]
        x = x.permute(0, 2, 1, 3, 4)                          # [B, T/2, C, H, W]
        return x


# ---------------------------------------------------------------------------
# Teacher
# ---------------------------------------------------------------------------

def load_teacher(checkpoint_path, num_frames=16, resolution=224, device='cpu'):
    """Load ViT-G teacher encoder from checkpoint. Returns frozen encoder."""
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


def extract_teacher_features(encoder, video_tensor):
    """
    Extract spatio-temporal features from the teacher.

    Args:
        encoder: ClipAggregation(ViT) from load_teacher()
        video_tensor: [B, C, F=16, H=224, W=224]
    Returns:
        [B, 8, 14, 14, 1408]
    """
    B, C, F, H, W = video_tensor.shape
    clips = [[video_tensor]]

    with torch.no_grad():
        outputs = encoder(clips)        # list of [B, 1568, 1408]

    feat = outputs[0]                   # [B, 1568, 1408]

    T_tokens = F // 2                   # 8  (tubelet_size=2)
    S_tokens = feat.size(1) // T_tokens # 196
    H_patches = int(S_tokens ** 0.5)    # 14
    W_patches = H_patches

    feat = feat.view(B, T_tokens, H_patches, W_patches, -1)  # [B, 8, 14, 14, 1408]
    return feat


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_distill_checkpoint(backbone, projector, temporal_agg, optimizer,
                            epoch, metrics, save_path):
    """Save Stage-1 distillation checkpoint."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        'backbone_state_dict': backbone.state_dict(),
        'projector_state_dict': projector.state_dict(),
        'temporal_aggregator_state_dict': temporal_agg.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'metrics': metrics,
        'meta': {
            'teacher_embed_dim': 1408,
            'student_in_channels': backbone.feature_channels,
            'spatial_size': backbone.feature_size,
            'temporal_tokens': 8,
            'model_name': backbone.model_name,
            'num_classes': backbone.num_classes,
        }
    }, save_path)


def load_distilled_backbone(weight_path, model_name="mobilenetv3_small_075",
                            num_classes=2, device='cpu'):
    """
    Load MobileNetV3Backbone from a Stage-1 distillation checkpoint.

    Returns (backbone, meta, load_info).
    load_info is a dict with keys: matched, extra, missing.
    If weight_path is empty or does not exist, returns model with random init.
    """
    load_info = {'matched': 0, 'extra': 0, 'missing': 0}

    if not weight_path or not os.path.isfile(weight_path):
        backbone = MobileNetV3Backbone(
            model_name=model_name,
            num_classes=num_classes,
        )
        load_info['missing'] = len(list(backbone.state_dict().keys()))
        return backbone, {}, load_info

    ckpt = torch.load(weight_path, map_location=device)
    meta = ckpt.get('meta', {})
    backbone = MobileNetV3Backbone(
        model_name=meta.get('model_name', model_name),
        num_classes=meta.get('num_classes', num_classes),
    )

    ckpt_state = ckpt.get('backbone_state_dict', {})
    if not ckpt_state:
        load_info['missing'] = len(list(backbone.state_dict().keys()))
        return backbone, meta, load_info

    model_keys = set(backbone.state_dict().keys())
    ckpt_keys = set(ckpt_state.keys())
    matched = model_keys & ckpt_keys
    extra = ckpt_keys - model_keys
    missing = model_keys - ckpt_keys

    backbone.load_state_dict(ckpt_state, strict=False)
    load_info = {'matched': len(matched), 'extra': len(extra), 'missing': len(missing)}
    print(f'loaded pretrained model: matched {len(matched)}, extra {len(extra)}, missing {len(missing)}')
    return backbone, meta, load_info
