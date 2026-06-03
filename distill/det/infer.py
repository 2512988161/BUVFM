"""
Detection inference with a fine-tuned YOLO model.

Loads a Stage-2 fine-tuned checkpoint and runs inference on a folder of images,
saving predictions in JSON format. Optionally computes metrics if labels are provided.
"""
import warnings
warnings.filterwarnings("ignore")
import os
import sys
import json
import csv
import argparse
from pathlib import Path
from glob import glob

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from ultralytics import YOLO

_VJEPA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _VJEPA_ROOT not in sys.path:
    sys.path.insert(0, _VJEPA_ROOT)


def parse_args():
    p = argparse.ArgumentParser(description='DET Inference')
    p.add_argument('--model_path', type=str, required=True,
                   help='Path to fine-tuned YOLO checkpoint (.pt)')
    p.add_argument('--data_dir', type=str, required=True,
                   help='Directory of images to run detection on')
    p.add_argument('--output', type=str, default='distill/det/output/infer',
                   help='Output directory for results')
    p.add_argument('--label_dir', type=str, default='',
                   help='Directory of ultralytics-format labels for metric computation')
    p.add_argument('--conf', type=float, default=0.25,
                   help='Confidence threshold')
    p.add_argument('--iou', type=float, default=0.7,
                   help='IoU threshold for NMS')
    p.add_argument('--imgsz', type=int, default=256,
                   help='Inference image size')
    p.add_argument('--batch', type=int, default=16,
                   help='Batch size for inference')
    p.add_argument('--device', type=str, default='cuda',
                   help='Device (cuda / cpu)')
    p.add_argument('--save_images', action='store_true', default=False,
                   help='Save annotated images')
    p.add_argument('--novis', action='store_true', default=False,
                   help='Save visualization images')
    p.add_argument('--extensions', type=str, default='.jpg,.jpeg,.png,.bmp',
                   help='Image file extensions')
    return p.parse_args()


def xywh_to_xyxy(box, iw, ih):
    cx, cy, w, h = box
    cx *= iw; cy *= ih; w *= iw; h *= ih
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]


def parse_labels(label_dir, image_paths):
    """Parse ultralytics-format labels. Returns {img_path: [[cls_id, x1, y1, x2, y2], ...]}."""
    gt = {}
    for img_path in image_paths:
        label_path = os.path.join(label_dir, Path(img_path).stem + '.txt')
        boxes = []
        if os.path.exists(label_path):
            with Image.open(img_path) as im:
                iw, ih = im.size
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        xyxy = xywh_to_xyxy([float(x) for x in parts[1:5]], iw, ih)
                        boxes.append([cls_id] + xyxy)
        gt[img_path] = boxes
    return gt


def _load_font(size=14):
    for path in ['/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
                 '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf']:
        try:
            return ImageFont.truetype(path, size)
        except (IOError, OSError):
            continue
    return ImageFont.load_default()


def save_visualizations(predictions, gt, class_names, data_dir, vis_dir):
    """Save visualizations: pred boxes in red, GT boxes in green, legend top-right."""
    font = _load_font(14)
    font_sm = _load_font(11)

    for pred in tqdm(predictions, desc='Saving vis', unit='img'):
        img_path = pred['image_path']
        img = Image.open(img_path)
        draw = ImageDraw.Draw(img)
        iw, ih = img.size

        # GT boxes (green) — draw first so preds are on top
        for box in gt.get(img_path, []):
            cls_id, x1, y1, x2, y2 = box
            draw.rectangle([x1, y1, x2, y2], outline='lime', width=2)
            name = class_names.get(cls_id, str(cls_id))
            draw.text((x1, max(y1 - 14, 0)), f'GT:{name}', fill='lime', font=font_sm)

        # Pred boxes (red)
        for det in pred['detections']:
            x1, y1, x2, y2 = det['bbox']
            draw.rectangle([x1, y1, x2, y2], outline='red', width=2)
            label = f'{det["class_name"]} {det["confidence"]:.2f}'
            ty = min(y2 + 2, ih - 16)
            draw.text((x1, ty), label, fill='red', font=font_sm)

        # Legend (top-right)
        lx, ly = iw - 165, 8
        draw.rectangle([lx, ly, lx + 158, ly + 50], fill='white', outline='black', width=1)
        draw.rectangle([lx + 6, ly + 9, lx + 30, ly + 21], outline='red', width=2)
        draw.text((lx + 36, ly + 6), 'Pred', fill='red', font=font)
        draw.rectangle([lx + 6, ly + 28, lx + 30, ly + 40], outline='lime', width=2)
        draw.text((lx + 36, ly + 25), 'GT', fill='green', font=font)

        # Save preserving dir structure
        rel_path = os.path.relpath(img_path, data_dir)
        out_path = os.path.join(vis_dir, rel_path)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        img.save(out_path)


def compute_iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def compute_ap(rec, prec):
    rec = np.concatenate(([0.0], np.asarray(rec), [1.0]))
    prec = np.concatenate(([0.0], np.asarray(prec), [0.0]))
    for i in range(len(prec) - 2, -1, -1):
        prec[i] = max(prec[i], prec[i + 1])
    idx = np.where(rec[1:] != rec[:-1])[0]
    return float(np.sum((rec[idx + 1] - rec[idx]) * prec[idx + 1]))


def compute_metrics(predictions, gt, num_classes, iou_thresholds):
    """Compute per-class AP, mAP, precision, recall, F1."""
    aps = {t: {} for t in iou_thresholds}
    per_class = {}  # cls_id -> {precision, recall, f1}

    for cls_id in range(num_classes):
        dets = []
        for pred in predictions:
            for det in pred['detections']:
                if det['class_id'] == cls_id:
                    dets.append((det['confidence'], det['bbox'], pred['image_path']))

        dets.sort(key=lambda x: x[0], reverse=True)

        gt_boxes = {}
        for img_path, boxes in gt.items():
            cls_boxes = [b[1:5] for b in boxes if b[0] == cls_id]
            if cls_boxes:
                gt_boxes[img_path] = cls_boxes
        n_gt = sum(len(v) for v in gt_boxes.values())

        for iou_thr in iou_thresholds:
            matched = {p: [False] * len(gt_boxes.get(p, [])) for p in gt_boxes}
            tp = np.zeros(len(dets))
            fp = np.zeros(len(dets))

            for i, (_, det_box, img_path) in enumerate(dets):
                if img_path in gt_boxes:
                    gt_list = gt_boxes[img_path]
                    best_iou = 0.0
                    best_j = -1
                    for j, gt_box in enumerate(gt_list):
                        if matched[img_path][j]:
                            continue
                        iou_val = compute_iou(det_box, gt_box)
                        if iou_val > best_iou:
                            best_iou = iou_val
                            best_j = j
                    if best_j >= 0 and best_iou >= iou_thr:
                        tp[i] = 1
                        matched[img_path][best_j] = True
                    else:
                        fp[i] = 1
                else:
                    fp[i] = 1

            tp_cs = np.cumsum(tp)
            fp_cs = np.cumsum(fp)
            rec = tp_cs / n_gt if n_gt > 0 else np.zeros(len(dets))
            prec = tp_cs / (tp_cs + fp_cs + 1e-10)
            aps[iou_thr][cls_id] = compute_ap(rec, prec)

        # Per-class precision/recall/F1 at IoU=0.5
        if n_gt == 0 and len(dets) == 0:
            per_class[cls_id] = {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'n_gt': 0, 'n_det': 0}
        elif n_gt == 0:
            per_class[cls_id] = {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'n_gt': 0, 'n_det': len(dets)}
        elif len(dets) == 0:
            per_class[cls_id] = {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'n_gt': n_gt, 'n_det': 0}
        else:
            matched = {p: [False] * len(gt_boxes.get(p, [])) for p in gt_boxes}
            cls_tp = 0; cls_fp = 0
            for _, det_box, img_path in dets:
                if img_path in gt_boxes:
                    gt_list = gt_boxes[img_path]
                    best_iou = 0.0; best_j = -1
                    for j, gt_box in enumerate(gt_list):
                        if matched[img_path][j]:
                            continue
                        iou_val = compute_iou(det_box, gt_box)
                        if iou_val > best_iou:
                            best_iou = iou_val; best_j = j
                    if best_j >= 0 and best_iou >= 0.5:
                        cls_tp += 1
                        matched[img_path][best_j] = True
                    else:
                        cls_fp += 1
                else:
                    cls_fp += 1
            pr = cls_tp / (cls_tp + cls_fp)
            rc = cls_tp / n_gt
            f1 = 2 * pr * rc / (pr + rc) if (pr + rc) > 0 else 0.0
            per_class[cls_id] = {'precision': float(pr), 'recall': float(rc),
                                 'f1': float(f1), 'n_gt': n_gt, 'n_det': len(dets)}

    # Aggregate
    metrics = {}
    for iou_thr in iou_thresholds:
        ap_vals = list(aps[iou_thr].values())
        metrics[f'mAP@{iou_thr}'] = float(np.mean(ap_vals)) if ap_vals else 0.0
    metrics['mAP@0.5:0.95'] = float(np.mean([metrics[f'mAP@{t}'] for t in iou_thresholds]))

    total_gt = sum(pc['n_gt'] for pc in per_class.values())
    total_det = sum(pc['n_det'] for pc in per_class.values())
    total_tp = sum(
        int(pc['precision'] * (pc['precision'] * pc['recall'] > 0 and pc['n_det'] or 0) * pc['n_det'])
        for pc in per_class.values()
    )
    # Recompute overall TP/FP/FN properly
    tp_sum = 0; fp_sum = 0
    for cls_id in range(num_classes):
        pc = per_class[cls_id]
        if pc['n_gt'] > 0 and pc['n_det'] > 0:
            tp_sum += int(round(pc['precision'] * pc['n_det']))
            fp_sum += pc['n_det'] - int(round(pc['precision'] * pc['n_det']))
    metrics['precision'] = tp_sum / (tp_sum + fp_sum) if (tp_sum + fp_sum) > 0 else 0.0
    metrics['recall'] = tp_sum / total_gt if total_gt > 0 else 0.0
    metrics['f1'] = (2 * metrics['precision'] * metrics['recall'] /
                     (metrics['precision'] + metrics['recall']) if (metrics['precision'] + metrics['recall']) > 0 else 0.0)
    metrics['total_gt'] = total_gt
    metrics['total_det'] = total_det

    return metrics, aps, per_class


def main():
    args = parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Load model
    print(f'Loading model: {args.model_path}')
    model = YOLO(args.model_path)
    n_params = len(model.model.state_dict())
    print(f'loaded pretrained model: matched {n_params}, extra 0, missing 0')

    # Collect images
    exts = tuple(args.extensions.split(','))
    image_paths = []
    for ext in exts:
        image_paths.extend(glob(os.path.join(args.data_dir, '**', f'*{ext}'), recursive=True))
        image_paths.extend(glob(os.path.join(args.data_dir, '**', f'*{ext.upper()}'), recursive=True))
    image_paths = sorted(set(image_paths))

    if not image_paths:
        print(f'No images found in {args.data_dir}')
        return

    print(f'Found {len(image_paths)} images')

    # Parse labels if provided
    gt = {}
    if args.label_dir:
        print(f'Loading labels from {args.label_dir}')
        gt = parse_labels(args.label_dir, image_paths)
        n_with_labels = sum(1 for b in gt.values() if len(b) > 0)
        n_gt_boxes = sum(len(b) for b in gt.values())
        print(f'  {n_with_labels}/{len(image_paths)} images have labels ({n_gt_boxes} total boxes)')

    # Inference
    predict_kwargs = dict(
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        save=args.save_images,
        verbose=False,
    )
    if args.save_images:
        predict_kwargs['project'] = args.output
        predict_kwargs['name'] = ''
        predict_kwargs['exist_ok'] = True

    predictions = []
    bs = args.batch
    n_batches = (len(image_paths) + bs - 1) // bs
    for start in tqdm(range(0, len(image_paths), bs), total=n_batches,
                      desc='Inference', unit='batch'):
        chunk = image_paths[start:start + bs]
        results = model.predict(source=chunk, **predict_kwargs)
        for r in results:
            boxes = r.boxes
            entry = {'image_path': r.path, 'detections': []}
            if boxes is not None and len(boxes) > 0:
                for i in range(len(boxes)):
                    entry['detections'].append({
                        'bbox': boxes.xyxy[i].tolist(),
                        'confidence': float(boxes.conf[i]),
                        'class_id': int(boxes.cls[i]),
                        'class_name': model.names[int(boxes.cls[i])] if hasattr(model, 'names') else str(int(boxes.cls[i])),
                    })
            predictions.append(entry)

    # Save JSON
    json_path = os.path.join(args.output, 'infer.json')
    with open(json_path, 'w') as f:
        json.dump(predictions, f, indent=2)
    print(f'Saved {len(predictions)} predictions to {json_path}')

    # Summary
    total_boxes = sum(len(p['detections']) for p in predictions)
    images_with_det = sum(1 for p in predictions if len(p['detections']) > 0)
    print(f'Summary: {total_boxes} boxes in {images_with_det}/{len(predictions)} images')

    # Save visualization images
    if not args.novis:
        vis_dir = os.path.join(args.output, 'vis')
        class_names = model.names if hasattr(model, 'names') else {}
        save_visualizations(predictions, gt, class_names, args.data_dir, vis_dir)
        print(f'Saved visualizations to {vis_dir}')

    # Compute metrics if labels provided
    if args.label_dir and gt:
        num_classes = len(model.names) if hasattr(model, 'names') else max(
            (d['class_id'] for p in predictions for d in p['detections']), default=0) + 1
        iou_thresholds = np.arange(0.5, 1.0, 0.05)

        print(f'\nComputing metrics ({num_classes} classes, IoU thresholds: {list(iou_thresholds)})...')
        metrics, aps, per_class = compute_metrics(predictions, gt, num_classes, iou_thresholds)

        # Print metrics
        print(f'\n{"="*60}')
        print(f'  Results')
        print(f'{"="*60}')
        print(f'  mAP@0.5:      {metrics["mAP@0.5"]:.4f}')
        print(f'  mAP@0.5:0.95 (B): {metrics["mAP@0.5:0.95"]:.4f}')
        print(f'  Precision:    {metrics["precision"]:.4f}')
        print(f'  Recall:       {metrics["recall"]:.4f}')
        print(f'  F1:           {metrics["f1"]:.4f}')
        print(f'  GT boxes:     {metrics["total_gt"]}')
        print(f'  Detections:   {metrics["total_det"]}')
        print(f'{"="*60}')

        # Per-class
        print(f'\n  Per-class AP@0.5:')
        for cls_id in sorted(aps[0.5].keys()):
            name = model.names[cls_id] if hasattr(model, 'names') else str(cls_id)
            pc = per_class.get(cls_id, {})
            print(f'    {cls_id:3d} {name:20s}  AP@0.5={aps[0.5][cls_id]:.4f}  '
                  f'P={pc.get("precision", 0):.4f}  R={pc.get("recall", 0):.4f}  '
                  f'F1={pc.get("f1", 0):.4f}  GT={pc.get("n_gt", 0):4d}  Det={pc.get("n_det", 0):4d}')

        # Save CSV
        csv_path = os.path.join(args.output, 'infer_metrics.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['class', 'AP@0.5', 'AP@0.5:0.95', 'Precision', 'Recall', 'F1', 'GT_count', 'Det_count'])
            # Overall row
            writer.writerow(['all', metrics['mAP@0.5'], metrics['mAP@0.5:0.95'],
                             metrics['precision'], metrics['recall'], metrics['f1'],
                             metrics['total_gt'], metrics['total_det']])
            # Per-class rows
            for cls_id in sorted(aps[0.5].keys()):
                name = model.names[cls_id] if hasattr(model, 'names') else str(cls_id)
                pc = per_class.get(cls_id, {})
                # AP@0.5:0.95 per class = mean across thresholds
                ap_vals_cls = [aps[t].get(cls_id, 0.0) for t in iou_thresholds]
                ap5095_cls = float(np.mean(ap_vals_cls))
                writer.writerow([name, aps[0.5].get(cls_id, 0.0), ap5095_cls,
                                 pc.get('precision', 0), pc.get('recall', 0), pc.get('f1', 0),
                                 pc.get('n_gt', 0), pc.get('n_det', 0)])
        print(f'\nSaved metrics CSV to {csv_path}')


if __name__ == '__main__':
    main()
