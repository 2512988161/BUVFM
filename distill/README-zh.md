# Distill

ViT-G 教师 → 轻量学生模型，两阶段知识蒸馏。

---

## CLS — 分类蒸馏 (ViT-G → MobileNetV3-Small)

```
Stage 1: distill.py          Stage 2: finetune.py         Stage 3: infer.py
视频 → 逐帧学生特征            蒸馏 backbone → 分类头        图像分类推理
时序池化 → 与教师对齐           2cls 图像数据集微调            输出 CSV
```

| 权重文件 | 阶段 | 大小 | 说明 |
|---|---|---|---|
| `best_distill.pt` | Stage 1 | 5.8 MB | 蒸馏后的 MobileNetV3 backbone |
| `best_finetune_distilled.pt` | Stage 2 | 4.1 MB | 蒸馏 backbone 微调 |
| `best_finetune_scratch.pt` | Stage 2 | 4.1 MB | 从头训练微调（基线） |

### 数据集准备

**Stage 1** — VideoFolderDataset（类别子目录，每个目录下放 `.mp4`）：

```
videos_train/
├── class_0/
│   ├── video_001.mp4
│   └── ...
└── class_1/
    ├── video_001.mp4
    └── ...
```

**Stage 2 & 3** — ImageFolder（类别子目录，每个目录下放图像）：

```
2cls_classification/
├── train/
│   ├── class_0/   (图像)
│   └── class_1/   (图像)
└── val/
    ├── class_0/   (图像)
    └── class_1/   (图像)
```

### Stage 1 — 蒸馏

```bash
torchrun --nproc_per_node=4 distill/cls/distill.py \
    --teacher_ckpt './ckpts/vjepa_full/best_vjepa_model9639(paper).pt' \
    --video_dir /home/lx/dataset/videos_train \
    --batch_size 40 --epochs 50
```

### Stage 2 — 微调

```bash
torchrun --nproc_per_node=4 distill/cls/finetune.py \
    --distilled_ckpt distill/cls/output/ckpts/best_distill.pt \
    --train_dir /home/lx/dataset/2cls_classification/train \
    --output_dir distill/cls/output/ft-distill-0525 \
    --epochs 200 --batch_size 512
```

### Stage 3 — 推理

```bash
# 蒸馏模型
python distill/cls/infer.py \
    --model_path distill/cls/output/ckpts/best_finetune_distilled.pt \
    --data_dir /home/lx/dataset/2cls_classification/val \
    --output_csv distill/cls/output/infer/distilled/predictions.csv

# 从头训练模型
python distill/cls/infer.py \
    --model_path distill/cls/output/ckpts/best_finetune_scratch.pt \
    --data_dir /home/lx/dataset/2cls_classification/val \
    --output_csv distill/cls/output/infer/scratch/predictions.csv
```

---

## DET — 检测蒸馏 (ViT-G → YOLO11m)

```
Stage 1: train.py              Stage 1.1: finetune.py       Stage 2: infer.py
YOLO 训练 + hook 中间层特征     加载 backbone → 完整 YOLO    目标检测推理
与教师对齐（2D 伪视频）          neck+head 从头训练            输出 JSON
```

### 权重

| 权重文件 | 阶段 | 大小 | 说明 |
|---|---|---|---|
| `distilled.pt` | Stage 1 | 193 MB | 蒸馏后的 YOLO11m backbone |
| `ft-scratch.pt` | Stage 1.1 | 39 MB | 从头训练微调（基线） |

### 数据集准备

**Stage 1 & 1.1** — YOLO 格式，通过 `data.yaml` 配置：

```yaml
# data.yaml
path: /path/to/dataset_root
train: images/train2017
val: images/val2017
names:
  0: class_0
  1: class_1
```

目录结构：
```
dataset_root/
├── images/
│   ├── train2017/   (*.jpg)
│   └── val2017/     (*.jpg)
└── labels/
    ├── train2017/   (*.txt，每张图像一个)
    └── val2017/     (*.txt，每张图像一个)
```

标签 `.txt` 格式（YOLO 归一化）：
```
<class_id> <cx> <cy> <w> <h>
```

**Stage 2 (推理)** — 图像目录（递归搜索），可选 YOLO 格式标签目录用于计算指标。

### Stage 1 — 蒸馏

```bash
PYTHONPATH=$(pwd) python distill/det/train.py \
    --teacher_ckpt './ckpts/vjepa_full/best_vjepa_model9639(paper).pt' \
    --data /home/lx/dataset/6cls_detection/data.yaml \
    --model yolo11m.pt --imgsz 640 --batch 256 --epochs 100 \
    --output_dir distill/det/output/distill-cls-ckpt \
    --device 0,1,2,3
```

### Stage 1.1 — 从头微调

```bash
torchrun --master_port=12333 --nproc_per_node=4 distill/det/finetune.py \
    --distilled_ckpt 'yolo11m.pt' \
    --data /home/lx/dataset/6cls_detection/data.yaml \
    --exp_name finetune-s --imgsz 640 --batch 256
```

### Stage 2 — 推理

```bash
# 蒸馏模型
export modelpath=distill/det/output/ckpts/distilled.pt
export infername="distilled"
python distill/det/infer.py \
    --model_path $modelpath --output distill/det/output/infer/${infername} \
    --data_dir /home/lx/dataset/6cls_detection/images/val2017/ \
    --label_dir /home/lx/dataset/6cls_detection/labels/val2017/ \
    --conf 0.25 --imgsz 640

# 从头训练模型
export modelpath=distill/det/output/ckpts/ft-scratch.pt
export infername="ft-scratch"
python distill/det/infer.py \
    --model_path $modelpath --output distill/det/output/infer/${infername} \
    --data_dir /home/lx/dataset/6cls_detection/images/val2017/ \
    --label_dir /home/lx/dataset/6cls_detection/labels/val2017/ \
    --conf 0.25 --imgsz 640
```

---

## QC — 质量控制筛查

`quality_con.py` — 多 GPU 级联筛查，使用蒸馏模型（MobileNetV3 + YOLO）进行逐帧病灶检测和片段提取。

```
输入视频 → MobileNetV3（逐帧分类）→ YOLO（仅低置信度帧）→ 每个视频提取 6 个片段
```

### 用法

```bash
python distill/quality_con.py \
    --input_folder /path/to/videos \
    --yolo_model_path distill/det/output/ckpts/distilled.pt \
    --svm_model_path distill/cls/output/ckpts/best_finetune_distilled.pt \
    --num_gpus 4 --batch_size 16
```

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--input_folder` | (必填) | 视频目录（递归扫描） |
| `--yolo_model_path` | distilled.pt | 蒸馏后的 YOLO 检测器 |
| `--svm_model_path` | ckpt 路径 | 蒸馏后的 MobileNetV3 分类器 |
| `--num_gpus` | 4 | GPU 数量 |
| `--batch_size` | 16 | 每个 Worker 的批大小 |
| `--yolo_conf_threshold` | 0.5 | YOLO 置信度阈值 |
| `--consecutive_frames` | 8 | 触发片段保存的连续检测帧数 |
| `--output_json_name` | output/QC/... | 最终汇总 JSON |
| `--clip_save_dir` | output/QC/... | 提取片段保存目录 |
| `--video_output_dir` | (空) | 可选的标注视频输出 |
