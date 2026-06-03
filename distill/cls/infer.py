"""
Inference with a fine-tuned MobileNetV3 classification model.

Loads a Stage-2 checkpoint and runs inference on a folder of images,
saving per-image predictions to a CSV file.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import timm
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)

_VJEPA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _VJEPA_ROOT not in sys.path:
    sys.path.insert(0, _VJEPA_ROOT)

from distill.cls.buildmodel import MobileNetV3Backbone

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


def parse_args():
    p = argparse.ArgumentParser(description='CLS Inference')
    p.add_argument('--model_path', type=str, required=True,
                   help='Path to fine-tuned checkpoint')
    p.add_argument('--data_dir', type=str, required=True,
                   help='Directory of images to classify (ImageFolder layout)')
    p.add_argument('--output_csv', type=str, default='predictions.csv',
                   help='Output CSV path')
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--num_classes', type=int, default=2)
    p.add_argument('--model_name', type=str, default='mobilenetv3_small_075',
                   help='timm MobileNetV3 model name')
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--device', type=str, default='cuda')
    return p.parse_args()


def load_model(model_path, num_classes=2, model_name='mobilenetv3_small_075', device='cuda'):
    """Load a fine-tuned model from a Stage-2 checkpoint or raw timm state_dict."""
    ckpt = torch.load(model_path, map_location=device, weights_only=False)

    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        model = MobileNetV3Backbone(model_name=model_name, num_classes=num_classes)
        state_dict = ckpt['model_state_dict']
    else:
        state_dict = ckpt
        model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)

    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(state_dict.keys())
    matched = len(model_keys & ckpt_keys)
    extra = len(ckpt_keys - model_keys)
    missing = len(model_keys - ckpt_keys)
    model.load_state_dict(state_dict, strict=False)
    print(f'loaded pretrained model: matched {matched}, extra {extra}, missing {missing}')

    model = model.to(device)
    model.eval()
    return model


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Dataset
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    dataset = datasets.ImageFolder(root=args.data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                        pin_memory=True, shuffle=False)

    # Model
    model = load_model(args.model_path, num_classes=args.num_classes,
                       model_name=args.model_name, device=device)

    class_names = dataset.classes

    # Inference
    results = []
    sample_idx = 0
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc='Inference'):
            imgs = imgs.to(device)
            logits = model.forward_full(imgs) if hasattr(model, 'forward_full') else model(imgs)
            probs = F.softmax(logits, dim=1)

            for i in range(imgs.size(0)):
                img_path = dataset.samples[sample_idx][0]
                pred = probs[i].argmax().item()

                row = {'image_path': img_path, 'label': labels[i].item(),
                       'predicted_class': pred, 'confidence': probs[i][pred].item()}
                for j in range(probs.size(1)):
                    row[f'p{j}_{class_names[j]}'] = probs[i][j].item()
                results.append(row)
                sample_idx += 1

    df = pd.DataFrame(results)

    # ---- Metrics ----
    y_true = df['label'].values
    y_pred = df['predicted_class'].values
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    print(f'\nAccuracy:  {acc:.4f}')
    print(f'Precision: {prec:.4f}')
    print(f'Recall:    {rec:.4f}')
    print(f'F1-score:  {f1:.4f}')
    print(f'\nConfusion Matrix:')
    print(cm)
    print(f'\n{classification_report(y_true, y_pred, target_names=class_names, zero_division=0)}')

    df.to_csv(args.output_csv, index=False)
    print(f'Saved {len(df)} predictions to {args.output_csv}')


if __name__ == '__main__':
    main()
