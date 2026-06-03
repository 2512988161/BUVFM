"""
Stage 1 — Detection distillation: ViT-G teacher → YOLO11m backbone.

Integrates with ultralytics DetectionTrainer.  The teacher is a standard
ViT-G (via root buildmodel.py).  A hook captures YOLO backbone features,
a 1×1 Conv adapter aligns channels, and the combined loss is:

    total = yolo_loss + α·MSE(proj_student, teacher) + β·(1−cosine_similarity)

Backbone weights are saved separately after training for Stage-2 fine-tuning.
"""
import warnings
warnings.filterwarnings("ignore")
import os
import sys
import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

import ultralytics
from ultralytics.models.yolo.detect.train import DetectionTrainer

_VJEPA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _VJEPA_ROOT not in sys.path:
    sys.path.insert(0, _VJEPA_ROOT)

from distill.det.buildmodel import (
    FeatureHook,
    YOLOFeatureAdapter,
    load_teacher,
    extract_teacher_2d_features,
    get_yolo_backbone_channels,
)


# ---------------------------------------------------------------------------
# KD loss wrapper (non-invasive — replaces model.loss)
# ---------------------------------------------------------------------------

class KDLossWrapper:
    """
    Wraps the original YOLO loss function, adding a distillation term
    between projected student features and teacher spatial features.

    Heavy objects (teacher, adapter) are stored on the trainer and
    accessed via reference to avoid deepcopy issues during DDP setup.
    """

    def __init__(self, original_loss_fn, trainer,
                 alpha=1.0, beta=0.5, num_frames=2, teacher_res=224,
                 hook_layer=6):
        self.original_loss_fn = original_loss_fn
        self._trainer = trainer
        self.alpha = alpha
        self.beta = beta
        self.num_frames = num_frames
        self.teacher_res = teacher_res
        self.hook_layer = hook_layer

    def __deepcopy__(self, memo):
        """Shallow copy — trainer ref must survive ultralytics deepcopy of model."""
        memo[id(self)] = self
        return self

    def __getstate__(self):
        """Exclude trainer reference from pickling (torch.save on model)."""
        state = self.__dict__.copy()
        state['_trainer'] = None
        return state

    # def _extract_student_features(self, imgs):
    #     """Manually run backbone layers up to hook_layer — no persistent hook needed."""
    #     x = imgs
    #     for i in range(self.hook_layer + 1):
    #         x = self._trainer.model.model[i](x)
    #     return x
    def _extract_student_features(self, imgs):
        """Manually run backbone layers up to hook_layer — no persistent hook needed."""
        x = imgs
        
        # 兼容 DDP 模式：如果模型被 DDP 包装，则取 .module
        base_model = self._trainer.model
        if hasattr(base_model, 'module'):
            base_model = base_model.module
            
        for i in range(self.hook_layer + 1):
            x = base_model.model[i](x)
        return x
    def __call__(self, batch, preds=None):
        trainer = self._trainer
        # 1. Original YOLO loss
        if preds is None:
            yolo_loss, loss_items = self.original_loss_fn(batch)
            is_training = True
        else:
            yolo_loss, loss_items = self.original_loss_fn(batch, preds)
            is_training = False

        if not is_training:
            kd_zero = torch.zeros(2, device=yolo_loss.device)
            return yolo_loss, torch.cat([loss_items, kd_zero])

        # 2. Extract student features (manual forward, no hook)
        imgs = batch['img']  # [B, C, H, W] (YOLO-normalized)
        student_feat = self._extract_student_features(imgs)

        # 3. Project student features
        student_proj = trainer.kd_adapter(student_feat)

        # 4. Teacher features (2D image → 2-frame pseudo-video)
        with torch.no_grad():
            teacher_feat = extract_teacher_2d_features(
                trainer.teacher_encoder, imgs, num_frames=self.num_frames,
                teacher_res=self.teacher_res,
            )

        # 5. Spatial alignment (teacher → student resolution)
        if teacher_feat.shape[-2:] != student_proj.shape[-2:]:
            teacher_feat = F.interpolate(
                teacher_feat, size=student_proj.shape[-2:],
                mode='bilinear', align_corners=False,
            )

        # 6. KD loss
        mse_val = F.mse_loss(student_proj, teacher_feat)

        s_flat = student_proj.flatten(1)
        t_flat = teacher_feat.flatten(1)
        cos_val = (1.0 - F.cosine_similarity(s_flat, t_flat, dim=1)).mean()

        kd_loss = self.alpha * mse_val + self.beta * cos_val

        total_loss = yolo_loss + kd_loss

        return total_loss, torch.cat([loss_items,
                                      mse_val.detach().view(1),
                                      cos_val.detach().view(1)])


# ---------------------------------------------------------------------------
# Custom trainer
# ---------------------------------------------------------------------------

class KDDetectionTrainer(DetectionTrainer):
    """
    Extends ultralytics DetectionTrainer with knowledge-distillation support.

    Reads KD configuration from environment variables (set by the caller before
    trainer.train() spawns DDP subprocesses).
    """

    def setup_model(self):
        super().setup_model()

        rank = int(os.environ.get('LOCAL_RANK', 0))

        # Read KD config from env
        teacher_ckpt = os.environ.get('KD_TEACHER_CKPT', '')
        kd_alpha = float(os.environ.get('KD_ALPHA', '1.0'))
        kd_beta = float(os.environ.get('KD_BETA', '0.5'))
        num_frames = int(os.environ.get('KD_NUM_FRAMES', '2'))
        hook_layer = int(os.environ.get('KD_HOOK_LAYER', '6'))
        teacher_res = int(os.environ.get('KD_TEACHER_RES', '224'))

        # Teacher
        if rank == 0:
            print(f'Loading teacher from {teacher_ckpt}')
        self.teacher_encoder = load_teacher(
            teacher_ckpt, num_frames=num_frames, resolution=teacher_res,
            device=self.device,
        )

        # Auto-detect student channels
        student_ch = get_yolo_backbone_channels(
            self.model, hook_layer, imgsz=self.args.imgsz,
        )
        if rank == 0:
            print(f'YOLO backbone layer {hook_layer}: {student_ch} channels')

        # Adapter (stored on trainer, not model, to survive deepcopy during DDP setup)
        self.kd_adapter = nn.Conv2d(
            student_ch, 1408, kernel_size=1,
        ).to(self.device)

        # Replace loss
        original_loss = self.model.loss
        self.model.loss = KDLossWrapper(
            original_loss_fn=original_loss,
            trainer=self,
            alpha=kd_alpha,
            beta=kd_beta,
            num_frames=num_frames,
            teacher_res=teacher_res,
            hook_layer=hook_layer,
        )

    def save_model(self):
        """Save full YOLO checkpoint + backbone-only weights."""
        super().save_model()

        rank = int(os.environ.get('LOCAL_RANK', 0))
        if rank != 0:
            return

        # Only save distilled backbone on the final epoch
        if self.epoch != self.epochs - 1:
            return

        hook_layer = int(os.environ.get('KD_HOOK_LAYER', '6'))
        # Extract backbone weights (layers 0..hook_layer)
        backbone_state = {}
        
        # 兼容 DDP 模式
        base_model = self.model
        if hasattr(base_model, 'module'):
            base_model = base_model.module
            
        for i in range(hook_layer + 1):
            for name, param in base_model.model[i].named_parameters():
                backbone_state[f'model.{i}.{name}'] = param.data.cpu().clone()
                
        # # Extract backbone weights (layers 0..hook_layer)
        # backbone_state = {}
        # for i in range(hook_layer + 1):
        #     for name, param in self.model.model[i].named_parameters():
        #         backbone_state[f'model.{i}.{name}'] = param.data.cpu().clone()

        save_path = Path(self.save_dir) / 'distilled_backbone.pt'
        torch.save({
            'backbone_state_dict': backbone_state,
            'hook_layer': hook_layer,
            'teacher_embed_dim': 1408,
        }, save_path)
        print(f'Distilled backbone saved to {save_path}')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='DET Distillation: ViT-G → YOLO11m backbone')
    # Teacher
    p.add_argument('--teacher_ckpt', type=str, required=True,
                   help='Path to ViT-G checkpoint')
    # YOLO config
    p.add_argument('--data', type=str, required=True,
                   help='Path to YOLO dataset yaml')
    p.add_argument('--model', type=str, default='yolo11m.pt',
                   help='Base YOLO model to initialize from')
    # Training
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--imgsz', type=int, default=256)
    p.add_argument('--batch', type=int, default=128)
    p.add_argument('--device', type=str, default='0')
    # KD
    p.add_argument('--alpha', type=float, default=1.0,
                   help='MSE loss weight in KD')
    p.add_argument('--beta', type=float, default=0.5,
                   help='Cosine loss weight in KD')
    p.add_argument('--num_frames', type=int, default=2,
                   help='Temporal frames for teacher (minimum 2 for tubelet_size=2)')
    p.add_argument('--hook_layer', type=int, default=6,
                   help='YOLO backbone layer index to hook (default 6: C3k2 at stride 16)')
    p.add_argument('--teacher_res', type=int, default=224,
                   help='Teacher input resolution (must be divisible by 16)')
    # Output
    p.add_argument('--output_dir', type=str, default=None,
                   help='Output directory (default: distill/det/output/)')
    # Misc
    p.add_argument('--workers', type=int, default=8)
    return p.parse_args()


# def main():
#     args = parse_args()

#     output_dir = args.output_dir or os.path.join(os.path.dirname(__file__), 'output')
#     os.makedirs(output_dir, exist_ok=True)

#     # Pass KD config to spawned DDP subprocesses via environment variables
#     os.environ['KD_TEACHER_CKPT'] = args.teacher_ckpt
#     os.environ['KD_ALPHA'] = str(args.alpha)
#     os.environ['KD_BETA'] = str(args.beta)
#     os.environ['KD_NUM_FRAMES'] = str(args.num_frames)
#     os.environ['KD_HOOK_LAYER'] = str(args.hook_layer)
#     os.environ['KD_TEACHER_RES'] = str(args.teacher_res)

#     # Fix DDP import: ultralytics uses __module__ to generate the import statement;
#     # when running as __main__, it generates "from __main__ import ..." which fails
#     # in DDP subprocesses. Setting __module__ to the importable package path fixes it.
#     KDDetectionTrainer.__module__ = 'distill.det.distill'

#     trainer = KDDetectionTrainer(overrides={
#         'data': args.data,
#         'model': args.model,
#         'epochs': args.epochs,
#         'imgsz': args.imgsz,
#         'batch': args.batch,
#         'device': args.device,
#         'workers': args.workers,
#         'project': output_dir,
#         'name': '',
#         'exist_ok': True,
#     })
#     trainer.train()
def main():
    args = parse_args()

    output_dir = args.output_dir or os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(output_dir, exist_ok=True)

    # Pass KD config to spawned DDP subprocesses via environment variables
    os.environ['KD_TEACHER_CKPT'] = args.teacher_ckpt
    os.environ['KD_ALPHA'] = str(args.alpha)
    os.environ['KD_BETA'] = str(args.beta)
    os.environ['KD_NUM_FRAMES'] = str(args.num_frames)
    os.environ['KD_HOOK_LAYER'] = str(args.hook_layer)
    os.environ['KD_TEACHER_RES'] = str(args.teacher_res)

    # ---> ADD THIS BLOCK TO FIX DDP IMPORTS <---
    # Propagate the root path to DDP subprocesses via PYTHONPATH
    current_pythonpath = os.environ.get('PYTHONPATH', '')
    if _VJEPA_ROOT not in current_pythonpath.split(os.pathsep):
        os.environ['PYTHONPATH'] = f"{_VJEPA_ROOT}{os.pathsep}{current_pythonpath}".strip(os.pathsep)
    # -------------------------------------------

    # Fix DDP import: ultralytics uses __module__ to generate the import statement;
    # when running as __main__, it generates "from __main__ import ..." which fails
    # in DDP subprocesses. Setting __module__ to the importable package path fixes it.
    KDDetectionTrainer.__module__ = 'distill.det.train'

    trainer = KDDetectionTrainer(overrides={
        'data': args.data,
        'model': args.model,
        'epochs': args.epochs,
        'imgsz': args.imgsz,
        'batch': args.batch,
        'device': args.device,
        'workers': args.workers,
        'project': os.path.basename(output_dir),
        'name': '',
        'exist_ok': True,
        'save_period': 1,
    })
    trainer.train()

if __name__ == '__main__':
    main()
