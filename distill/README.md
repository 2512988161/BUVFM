# Distill

ViT-G teacher → lightweight student model, two-stage knowledge distillation.

---

## CLS — Classification Distillation (ViT-G → MobileNetV3-Small)

```
Stage 1: distill.py          Stage 2: finetune.py         Stage 3: infer.py
Video → per-frame student     Distilled backbone →         Image classification
features, temporal pool       classification head,         inference, output CSV
→ align with teacher          2cls image dataset
```

| Checkpoint | Stage | Size | Description |
|---|---|---|---|
| `best_distill.pt` | Stage 1 | 5.8 MB | Distilled MobileNetV3 backbone |
| `best_finetune_distilled.pt` | Stage 2 | 4.1 MB | Fine-tuned from distilled backbone |
| `best_finetune_scratch.pt` | Stage 2 | 4.1 MB | Fine-tuned from scratch (baseline) |

### Dataset Preparation

**Stage 1** — VideoFolderDataset (class subdirs, each containing `.mp4`):

```
videos_train/
├── class_0/
│   ├── video_001.mp4
│   └── ...
└── class_1/
    ├── video_001.mp4
    └── ...
```

**Stage 2 & 3** — ImageFolder (class subdirs, each containing images):

```
2cls_classification/
├── train/
│   ├── class_0/   (images)
│   └── class_1/   (images)
└── val/
    ├── class_0/   (images)
    └── class_1/   (images)
```

### Stage 1 — Distillation

```bash
torchrun --nproc_per_node=4 distill/cls/distill.py \
    --teacher_ckpt './ckpts/vjepa_full/best_vjepa_model9639(paper).pt' \
    --video_dir /path/to/videos_train \
    --batch_size 40 --epochs 50
```

### Stage 2 — Fine-tuning

```bash
torchrun --nproc_per_node=4 distill/cls/finetune.py \
    --distilled_ckpt distill/cls/output/ckpts/best_distill.pt \
    --train_dir /path/to/2cls_classification/train \
    --output_dir distill/cls/output/ft-distill-0525 \
    --epochs 200 --batch_size 512
```

### Stage 3 — Inference

```bash
# Distilled model
python distill/cls/infer.py \
    --model_path distill/cls/output/ckpts/best_finetune_distilled.pt \
    --data_dir /path/to/2cls_classification/val \
    --output_csv distill/cls/output/infer/distilled/predictions.csv

# From-scratch model
python distill/cls/infer.py \
    --model_path distill/cls/output/ckpts/best_finetune_scratch.pt \
    --data_dir /path/to/2cls_classification/val \
    --output_csv distill/cls/output/infer/scratch/predictions.csv
```

---

## DET — Detection Distillation (ViT-G → YOLO11m)

```
Stage 1: train.py               Stage 1.1: finetune.py       Stage 2: infer.py
YOLO training + hook             Load backbone → full YOLO,   Object detection
intermediate features,           train neck+head from         inference, output JSON
align with teacher (2D pseudo    scratch
video)
```

### Checkpoints

| Checkpoint | Stage | Size | Description |
|---|---|---|---|
| `distilled.pt` | Stage 1 | 193 MB | Distilled YOLO11m backbone |
| `ft-scratch.pt` | Stage 1.1 | 39 MB | Fine-tuned from scratch (baseline) |

### Dataset Preparation

**Stage 1 & 1.1** — YOLO format, configured via `data.yaml`:

```yaml
# data.yaml
path: /path/to/dataset_root
train: images/train2017
val: images/val2017
names:
  0: class_0
  1: class_1
```

Directory layout:
```
dataset_root/
├── images/
│   ├── train2017/   (*.jpg)
│   └── val2017/     (*.jpg)
└── labels/
    ├── train2017/   (*.txt, one per image)
    └── val2017/     (*.txt, one per image)
```

Each label `.txt` (YOLO normalized format):
```
<class_id> <cx> <cy> <w> <h>
```

**Stage 2 (infer)** — Images directory (recursive glob), optional YOLO-format label directory for metrics.

### Stage 1 — Distillation

```bash
PYTHONPATH=$(pwd) python distill/det/train.py \
    --teacher_ckpt './ckpts/vjepa_full/best_vjepa_model9639(paper).pt' \
    --data /path/to/data.yaml \
    --model yolo11m.pt --imgsz 640 --batch 256 --epochs 100 \
    --output_dir distill/det/output/distill-cls-ckpt \
    --device 0,1,2,3
```

### Stage 1.1 — Fine-tuning from scratch

```bash
torchrun --master_port=12333 --nproc_per_node=4 distill/det/finetune.py \
    --distilled_ckpt 'yolo11m.pt' \
    --data /path/to/data.yaml \
    --exp_name finetune-s --imgsz 640 --batch 256
```

### Stage 2 — Inference

```bash
# Distilled model
export modelpath=distill/det/output/ckpts/distilled.pt
export infername="distilled"
python distill/det/infer.py \
    --model_path $modelpath --output distill/det/output/infer/${infername} \
    --data_dir /path/to/dataset_root/images/val2017/ \
    --label_dir /path/to/dataset_root/labels/val2017/ \
    --conf 0.25 --imgsz 640

# From-scratch model
export modelpath=distill/det/output/ckpts/ft-scratch.pt
export infername="ft-scratch"
python distill/det/infer.py \
    --model_path $modelpath --output distill/det/output/infer/${infername} \
    --data_dir /path/to/dataset_root/images/val2017/ \
    --label_dir /path/to/dataset_root/labels/val2017/ \
    --conf 0.25 --imgsz 640
```

---

## QC — Quality Control Screening

`distill/quality_con.py` — Multi-GPU cascaded screening using distilled models (MobileNetV3 + YOLO) for frame-level lesion detection and clip extraction.

```
Input video → MobileNetV3 (frame classification) → YOLO (low-confidence frames only) → 6 clips per video
```

### Usage

```bash
python distill/quality_con.py \
    --input_folder /path/to/videos \
    --yolo_model_path distill/det/output/ckpts/distilled.pt \
    --svm_model_path distill/cls/output/ckpts/best_finetune_distilled.pt \
    --num_gpus 4 --batch_size 16
```

| Argument | Default | Description |
|---|---|---|
| `--input_folder` | (required) | Video directory (recursive scan) |
| `--yolo_model_path` | distilled.pt | Distilled YOLO detector |
| `--svm_model_path` | ckpt path | Distilled MobileNetV3 classifier |
| `--num_gpus` | 4 | Number of GPUs |
| `--batch_size` | 16 | Batch size per worker |
| `--yolo_conf_threshold` | 0.5 | YOLO confidence threshold |
| `--consecutive_frames` | 8 | Consecutive frames to trigger clip save |
| `--output_json_name` | output/QC/... | Final aggregated JSON |
| `--clip_save_dir` | output/QC/... | Extracted clips directory |
| `--video_output_dir` | (empty) | Optional annotated video output |
