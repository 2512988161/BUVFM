<div align="center">
<h1>BUVFM — Breast Ultrasound Video Foundation Model</h1>
<!-- <a href="http://arxiv.org/abs/2509.11752"><img src='https://img.shields.io/badge/arXiv-Preprint-red' alt='Paper PDF'></a> -->
<a href='https://huggingface.co/xenosscu/BUVFM'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Weights-blue'></a>
<a href='https://2512988161.github.io/nmODE-V-pp/'><img src='https://img.shields.io/badge/Homepage-green' alt='Homepage'></a>
<a href='http://buvfm.machineilab.org/'><img src='https://img.shields.io/badge/Demopage-blue' alt='Demopage'></a>
<a href='https://github.com/2512988161/BUVFM'><img src='https://img.shields.io/badge/Github-red' alt='Github'></a>
</div>

BUVFM is the first breast ultrasound video foundation model, pretrained on 423K frames and 84K video sequences via self-supervised learning and fine-tuned for BI-RADS risk stratification. It powers AIBS, a cloud-edge system that enables non-physician ultrasound acquisition with real-time quality control, and achieves 95.42% macro sensitivity and 95.60% macro specificity across internal and 11 external cohorts.

## Table of Contents

- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Dataset Preparation](#dataset-preparation)
- [Checkpoint Preparation](#checkpoint-preparation)
- [Pretraining](#pretraining)
- [Training](#training)
- [Knowledge Distillation](#knowledge-distillation)
- [Inference & Evaluation](#inference--evaluation)
- [Demo Applications](#demo-applications)
- [Results](#results)
- [License](#license)

## Roadmap

- [x] **VJEPA fine-tuning** — Downstream BI-RADS classification on ultrasound videos with ViT-Giant backbone.
- [x] **Inference & evaluation** — Multi-GPU distributed inference, t-SNE/PCA visualization, speckle noise robustness testing.
- [x] **QC screening pipeline** — MobileNetV3 + YOLO frame-level lesion screening and clip extraction.
- [x] **2-Stage demo** — Gradio web app: Stage 1 (MobileNetV3+YOLO screening) → Stage 2 (VJEPA2+Grad-CAM risk grading).
- [x] **Hugging Face Hub release** — Pretrained weights, fine-tuned checkpoints, and QC models on [Hugging Face](https://huggingface.co/xenosscu/BUVFM).
- [x] **VideoMAE (MAE)** — VideoMAEv2-style masked autoencoder with VJEPA ViT-g encoder and shallow Transformer decoder. Tube masking (90%) + running-cell masking (50%), per-patch normalized MSE loss.
- [x] **VideoVAEPlus (VAE)** — ViT encoder → 2+1D Conv decoder with temporal compression + 3D PatchGAN discriminator. L1 + LPIPS + KL + adversarial loss.
- [x] **Checkpoint conversion** — `pretraining/convert_checkpoint.py` converts MAE/VAE checkpoints to `build_model()`-compatible format for downstream fine-tuning.
- [ ] **BUVFM pretraining** — Pretrain a unified BUVFM foundation model on large-scale ultrasound video data.
- [x] **Knowledge distillation** — Distill the ViT-Giant model to smaller architectures for efficient deployment.

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download checkpoints

```bash
python download_ckpt.py
```

Downloads pretrained weights, fine-tuned checkpoints, and QC models from [Hugging Face Hub](https://huggingface.co/xenosscu/BUVFM) to `./ckpts/` and `./QC/`.

### 3. Prepare your dataset

Organize videos in an `ImageFolder`-style directory — each subfolder named `class_N` holds videos of that class:

```
dataset/
├── train/
│   ├── class_0/
│   ├── class_1/
│   └── class_2/
└── val/
    ├── class_0/
    ├── class_1/
    └── class_2/
```

Supported formats: `.mp4`, `.avi`, `.mov`, `.mkv`. See [Dataset Preparation](#dataset-preparation) for parameter details.

### 4. Fine-tune the pretrained model

```bash
torchrun --master_port=29511 --nproc_per_node=2 train.py \
  --train_dir /path/to/dataset/train \
  --val_dir /path/to/dataset/val \
  --pretrained_ckpt ./ckpts/vjepa-nmode-pretrain.pt \
  --exp_name exp1
```

See [Training](#training) for all arguments (freeze backbone, learning rate, model variants, etc.).

### 5. Run inference

```bash
export TRPORT=29599
export CKPT=/path/to/ckpt
export VAL_DIR='/path/to/val/dir'
export OUT='test'
export EXTRA=""
export FS=2
export FPC=16
torchrun --master_port=$TRPORT --nproc_per_node=4 inference_ddp_old.py \
    --val_dir $VAL_DIR --checkpoint $CKPT --output ./output/$OUT.csv \
    $EXTRA --frame_step $FS --frames_per_clip $FPC
```

See [Inference & Evaluation](#inference--evaluation) for more options.


## Project Structure

```
BUVFM/
├── download_ckpt.py           # Download all model weights from Hugging Face Hub
├── train.py                  # Fine-tuning script
├── inference_ddp_old.py      # Multi-GPU distributed inference 
├── test.py                   # Arbitrary-dir inference (hardcoded config)
├── app.py                    # Gradio 2-stage pipeline demo (port 9530)
├── pipeline_utils.py         # 2-stage pipeline: MobileNet+YOLO + VJEPA2+Grad-CAM
├── buildmodel.py             # Model construction & checkpoint loading
├── mydataset.py              # VideoFolderDataset, InferenceVideoDataset, etc.
├── utils.py                  # Data augmentation & video transforms
├── cluster_videos_nocohen.py # t-SNE / PCA feature embedding visualization
├── src/
│   ├── models/
│   │   ├── vision_transformer.py   # ViT backbone (giant/large/huge/tiny/...)
│   │   ├── attentive_pooler.py     # Attentive classifier head
│   │   ├── ac_predictor.py         # Action-conditioned predictor
│   │   └── utils/                  # modules, patch_embed, pos_embs
│   ├── masks/                      # Tube/block masking for VJEPA pretraining
│   ├── datasets/                   # Video dataset utils & transforms
│   │   └── utils/
│   │       ├── video/              # RandAugment, RandErase, volume transforms
│   │       ├── weighted_sampler.py
│   │       ├── worker_init_fn.py
│   │       └── dataloader.py
│   └── utils/                      # Distributed, logging, schedulers, checkpoint loader
├── pretraining/
│   ├── run_pretrain_mae.py         # VideoMAEv2-style MAE pretraining entry
│   ├── run_pretrain_vae.py         # VideoVAEPlus pretraining entry
│   ├── convert_checkpoint.py       # Pretrain ckpt → build_model format
│   ├── prepare_data.py             # Generate annotation files from raw videos
│   └── methods/                    # MAE/VAE modeling, datasets, engines, masking
├── distill/
│   ├── README.md                   # Distillation usage guide
│   ├── cls/                        # Classification distillation (ViT-G → MobileNetV3)
│   │   ├── distill.py              # Stage 1 — feature alignment on videos
│   │   ├── finetune.py             # Stage 2 — 2D image classification fine-tuning
│   │   └── infer.py                # Stage 3 — inference → CSV
│   └── det/                        # Detection distillation (ViT-G → YOLO11m)
│       ├── train.py                # Stage 1 — YOLO training + feature alignment
│       ├── finetune.py             # Stage 1.1 — full YOLO fine-tuning
│       └── infer.py                # Stage 2 — inference → JSON
├── QC/
│   ├── stage1.py                   # Standalone MobileNet+YOLO screener
│   ├── quality_con_clip_save_ed5.py# Multi-GPU batch clip extraction
│   ├── best_640_s_60e(2).pt        # YOLO detector weights
│   └── mobilenetv3_small_075_yl_241222(3).pth  # MobileNetV3 classifier weights
├── ckpts/
│   ├── vjepa-nmode-pretrain.pt     # Standard pretrained weights (fine-tuning start)
│   └── vjepa_full/                 # Fine-tuned checkpoints + eval CSVs
├── assets/                         # 16 example videos (4 per category: Class0/1/2/NO)
└── output/                         # Inference result CSVs & t-SNE outputs

```



## Dataset Preparation

Organize videos in an `ImageFolder`-style directory structure — each subfolder named `class_N` holds videos of that class:

```
dataset/
├── train/
│   ├── class_0/
│   │   ├── video_001.mp4
│   │   └── ...
│   ├── class_1/
│   │   └── ...
│   └── class_2/
│       └── ...
└── val/
    ├── class_0/
    ├── class_1/
    └── class_2/
```

Supported formats: `.mp4`, `.avi`, `.mov`, `.mkv`

Key dataset parameters:

| Parameter | Default | Description |
|---|---|---|
| `frames_per_clip` | 16 | Frames sampled per clip |
| `frame_step` | 2 | Step between sampled frames |
| `num_clips` | 1 | Number of clips per video |
| `random_clip_sampling` | True/False | Random clip start (train) vs. center (val) |

> **Note**: `train.py` applies class-0 undersampling to handle class imbalance (capped at 10,000), and use weighted loss functions.

## Checkpoint Preparation

### Download Checkpoints from Hugging Face Hub

Use `download_ckpt.py` to download all model weights (pretrained checkpoint, fine-tuned checkpoints, and QC models) from [xenosscu/BUVFM](https://huggingface.co/xenosscu/BUVFM) on Hugging Face Hub:

```bash
python download_ckpt.py
```

This downloads the full model repository to the current directory, preserving the expected directory structure (`ckpts/`, `QC/`).

### Pretrained Starting Point

Fine-tuning starts from `./ckpts/vjepa-nmode-pretrain.pt`. Alternative pretrained checkpoints are available under `ckpts/` (e.g., MAE-pretrained `vitg_mae_e1.pt`, VAE-pretrained `vitg_vae_e1.pt`, ViT-Large `vjepa-vitl.pt`). See the [Pretraining](#pretraining) section for how these are produced.

### Fine-tuned Checkpoints

Fine-tuned checkpoints are saved under `ckpts/vjepa_full/` (or variant directories like `vjepa_full_maee1`, `vjepa_full_newval`, etc.). Naming convention: `best_vjepa_model.pt` (best by validation accuracy).




## Pretraining

See `pretraining/README.md` for full details.


## Training

All training starts from a pretrained VJEPA checkpoint (`--pretrained_ckpt` is required). The standard starting point is `./ckpts/vjepa-nmode-pretrain.pt`.

### Fine-tuning (`train.py`)

Uses weighted `CrossEntropyLoss` with class-0 undersampling (capped at 10,000 samples).

```bash
torchrun --nproc_per_node=4 train.py \
    --pretrained_ckpt ./ckpts/vjepa-nmode-pretrain.pt \
    --batch_size 16 \
    --lr 5e-5 \
    --epochs 50
```


### Freeze Backbone

Train the classifier head only while keeping the encoder frozen:

```bash
torchrun --nproc_per_node=4 train.py \
    --pretrained_ckpt ./ckpts/vjepa-nmode-pretrain.pt \
    --batch_size 16 \
    --lr 5e-5 \
    --freeze_backbone
```

### Training Arguments

| Argument | Default | Description |
|---|---|---|
| `--pretrained_ckpt` | (required) | Path to pretrained checkpoint |
| `--batch_size` | 4 | Batch size per GPU |
| `--lr` | 5e-5 | Learning rate |
| `--epochs` | 50 | Number of training epochs |
| `--num_frames` | 16 | Frames per clip |
| `--model_name` | `vit_giant_xformers` | Backbone: `vit_giant_xformers`, `vit_large`, `vit_huge` |
| `--freeze_backbone` | False | Freeze encoder, train classifier only |
| `--exp_name` | None | Experiment name suffix for log/ckpt directories |
| `--train_dir` | `/home/lx/alg/videos_train` | Training data directory |
| `--val_dir` | `/home/lx/alg/videos_val` | Validation data directory |


### Training Output

- **Logs**: `./output/logs_vjepa/vjepa_full.log` (or `vjepa_frozen.log` if `--freeze_backbone`)
- **Checkpoints**: `./ckpts/vjepa_full/best_vjepa_model.pt` (best by val accuracy)
- **Eval CSV**: `./ckpts/vjepa_full/best_vjepa_eval.csv` (per-video predictions)

## Knowledge Distillation for Edge-Side Quality Control Model

See [distill/README.md](distill/README.md) for details.

## Inference & Evaluation

### Multi-GPU Distributed Inference (Primary)

Use `inference_ddp_old.py` with `torchrun` for parallel inference across GPUs:

```bash
export TRPORT=29599
export CKPT=/path/to/ckpt
export VAL_DIR='/path/to/val/dir'
export OUT='test'
export EXTRA=""
export FS=2
export FPC=16
torchrun --master_port=$TRPORT --nproc_per_node=4 inference_ddp_old.py \
    --val_dir $VAL_DIR --checkpoint $CKPT --output ./output/$OUT.csv \
    $EXTRA --frame_step $FS --frames_per_clip $FPC
```

| Argument | Default | Description |
|---|---|---|
| `--checkpoint` | (see code) | Path to fine-tuned checkpoint |
| `--val_dir` | (required) | One or more validation directories |
| `--output` | `./output/...` | Output CSV path |
| `--model_name` | `vit_giant_xformers` | Backbone variant |
| `--frame_step` | 2 | Step between sampled frames |
| `--frames_per_clip` | 16 | Frames sampled per clip |
| `--restore_true` | False | Run robust evaluation across speckle noise ratios (0.05–0.95) |

Each rank writes a temp CSV; rank 0 merges and deduplicates. Supports resume (skips already-processed videos if the output CSV exists).

### Arbitrary Directory Inference (no labels required)
```bash
python test.py
```
Edit the following variables in `test.py` before running:
- `checkpoint_path` — path to the fine-tuned checkpoint
- `input_dir` — path to directory containing videos (recursive scan)
- `output_csv` — output CSV path

## Demo

### 2-Stage Pipeline Demo

A web-based interactive demo implementing a cascaded 2-stage inference pipeline:

```
Video Input → [Stage 1: Screening] → [Stage 2: Risk Grading]
                    │                         │
          MobileNetV3 + YOLO           VJEPA2 ViT-Giant
          No lesions? → STOP           3-class probs + Grad-CAM
```

#### Running the Demo

```bash
pip install gradio plotly timm ultralytics
python app.py
```

Open http://localhost:9530 in your browser, or visit the online demo at [http://buvfm.machineilab.org/](http://buvfm.machineilab.org/). Upload a video or click an example (16 videos across 4 categories: Class 0 / Class NO / Class 1 / Class 2). The model is loaded on the first pipeline run.

Pipeline logic and model interfaces (MobileNetV3, YOLO, VJEPA2, Grad-CAM) are in `pipeline_utils.py`.

### Hugging Face Space
The Gradio demo in this repository is deployed at [http://buvfm.machineilab.org/](http://buvfm.machineilab.org/).

#### Local run
```bash
pip install gradio plotly timm ultralytics
pip install -r requirements.txt
CUDA_VISIBLE_DEVICES=1 python app.py
```
Optional environment variables:
```bash
export BUVFM_CHECKPOINT_PATH=/absolute/path/to/checkpoint.pt
export BUVFM_ASSETS_DIR=/absolute/path/to/assets
export BUVFM_MAX_FILE_SIZE_MB=100
```

## License
Licensed under the MIT License.
