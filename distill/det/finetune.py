"""
Stage 2 — Detection fine-tuning.

Loads a distilled YOLO backbone from Stage 1, then fine-tunes the full YOLO
model (backbone + neck + head) on a 2D detection dataset (YOLO format).

Optionally freeze the backbone to train only the neck and detection head.
"""

import os
import sys
import argparse
from pathlib import Path

import torch
from ultralytics import YOLO
from ultralytics.models.yolo.detect.train import DetectionTrainer

_VJEPA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _VJEPA_ROOT not in sys.path:
    sys.path.insert(0, _VJEPA_ROOT)

from distill.det.buildmodel import load_yolo_distilled_backbone


# ---------------------------------------------------------------------------
# Custom trainer with backbone-freezing support
# ---------------------------------------------------------------------------

class FinetuneDetectionTrainer(DetectionTrainer):
    """
    Extends DetectionTrainer to optionally freeze backbone layers after loading
    distilled weights.
    """

    def setup_model(self):
        super().setup_model()

        freeze_backbone = os.environ.get('FINETUNE_FREEZE_BACKBONE', '0') == '1'
        if not freeze_backbone:
            return

        # Freeze backbone layers 0–10 (standard YOLO11m backbone)
        freeze_until = int(os.environ.get('FINETUNE_FREEZE_UNTIL', '10'))
        rank = int(os.environ.get('LOCAL_RANK', '0'))

        if rank == 0:
            print(f'Freezing backbone layers 0–{freeze_until}')

        for i in range(freeze_until + 1):
            layer = self.model.model[i]
            for p in layer.parameters():
                p.requires_grad = False
            layer.eval()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='DET Fine-tuning on 2D detection data')
    # Checkpoint
    p.add_argument('--distilled_ckpt', type=str, required=True,
                   help='Path to Stage-1 distilled backbone checkpoint')
    # YOLO config
    p.add_argument('--data', type=str, required=True,
                   help='Path to YOLO dataset yaml')
    p.add_argument('--model', type=str, default='yolo11m.pt',
                   help='Base YOLO model to initialize neck+head from')
    # Training
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--imgsz', type=int, default=256)
    p.add_argument('--batch', type=int, default=128)
    p.add_argument('--device', type=str, default='0,1,2,3')
    # Freeze
    p.add_argument('--freeze_backbone', action='store_true',
                   help='Freeze distilled backbone, train only neck+head')
    p.add_argument('--freeze_until', type=int, default=10,
                   help='Last backbone layer index (default 10 for YOLO11m)')
    # Output
    p.add_argument('--output_dir', type=str, default='distill/det/output/',
                   help='Output directory (default: distill/det/output/)')
    p.add_argument('--exp_name', type=str, default='finetune',
                   help='Experiment name for output subdirectory')
    # Misc
    p.add_argument('--workers', type=int, default=8)
    return p.parse_args()


def main():
    args = parse_args()

    output_dir = os.path.join(args.output_dir,args.exp_name)
    os.makedirs(output_dir, exist_ok=True)

    # Load base YOLO model
    print(f'Loading base model: {args.model}')
    model = YOLO(args.model)

    # Load distilled backbone weights
    model = load_yolo_distilled_backbone(model, args.distilled_ckpt)

    # Save the initialized model for the trainer to resume from
    init_path = os.path.join(output_dir, 'init_distilled.pt')
    model.save(init_path)

    # Pass freeze config via env (DetectionTrainer spawns subprocesses)
    os.environ['FINETUNE_FREEZE_BACKBONE'] = '1' if args.freeze_backbone else '0'
    os.environ['FINETUNE_FREEZE_UNTIL'] = str(args.freeze_until)

    trainer = FinetuneDetectionTrainer(overrides={
        'data': args.data,
        'model': init_path,
        'epochs': args.epochs,
        'imgsz': args.imgsz,
        'batch': args.batch,
        'device': args.device,
        'workers': args.workers,
        # 'project': output_dir,
        'name': args.exp_name,
        'exist_ok': True,
    })
    trainer.train()

    print(f'Fine-tuning complete. Results saved to {output_dir}/')


if __name__ == '__main__':
    main()
