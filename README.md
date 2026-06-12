<div align="center">
<h1>BUVFM — Breast Ultrasound Video Foundation Model</h1>
<!-- <a href="http://arxiv.org/abs/2509.11752"><img src='https://img.shields.io/badge/arXiv-Preprint-red' alt='Paper PDF'></a> -->
<a href='https://huggingface.co/xenosscu/BUVFM'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Weights-blue'></a>
<a href='https://huggingface.co/datasets/xenosscu/BUVFM'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-yellow'></a>
<a href='https://2512988161.github.io/nmODE-V-pp/'><img src='https://img.shields.io/badge/Homepage-green' alt='Homepage'></a>
<a href='http://buvfm.machineilab.org/'><img src='https://img.shields.io/badge/Demopage-blue' alt='Demopage'></a>
<a href='https://github.com/2512988161/BUVFM'><img src='https://img.shields.io/badge/Github-red' alt='Github'></a>
</div>

BUVFM is the first breast ultrasound video foundation model, pretrained on 423K frames and 84K video sequences via self-supervised learning and fine-tuned for BI-RADS risk stratification. It powers AIBS, a cloud-edge system that enables non-physician ultrasound acquisition with real-time quality control, and achieves 95.42% macro sensitivity and 95.60% macro specificity across internal and 11 external cohorts.


- [x] **VJEPA fine-tuning** — Downstream BI-RADS classification on ultrasound videos with ViT-Giant backbone.
- [x] **Inference & evaluation** — Multi-GPU distributed inference, t-SNE/PCA visualization, speckle noise robustness testing.
- [x] **QC screening pipeline** — MobileNetV3 + YOLO frame-level lesion screening and clip extraction.
- [x] **2-Stage demo** — Gradio web app: Stage 1 (MobileNetV3+YOLO screening) → Stage 2 (VJEPA2+Grad-CAM risk grading).
- [x] **Hugging Face Hub release** — Pretrained weights, fine-tuned checkpoints, and QC models on [Hugging Face](https://huggingface.co/xenosscu/BUVFM).
- [x] **Checkpoint conversion** — `pretraining/convert_checkpoint.py` converts MAE/VAE checkpoints to `build_model()`-compatible format for downstream fine-tuning.
- [x] **BUVFM pretraining** — Pretrain a unified BUVFM foundation model on large-scale ultrasound video data.
- [x] **Knowledge distillation** — Distill the ViT-Giant model to smaller architectures for efficient deployment.

## Quick Start
<details>
<summary>Show/Hide</summary>
### 1. Install dependencies

```bash
conda create -n buvfm python=3.10.9 -y
conda activate buvfm
pip install -r requirements.txt
```

> **Typical install time**: ~5–10 minutes on a normal desktop computer with broadband internet. Model weight download time (step 2) depends on network speed; weights total approximately 20.9 GB.

### 2. Download demo datasets & checkpoints

```bash
HF_TOKEN=your-huggingface-token python download_dataset_ckpt.py
```

Downloads all demo datasets to `./dataset/` and all model weights to `./` from [xenosscu/BUVFM](https://huggingface.co/xenosscu/BUVFM) on Hugging Face Hub.

#### Datasets

| Name | Save Path | Description |
|------|-----------|-------------|
| `orilong` | `dataset/orilong/` | Original long ultrasound videos |
| `videos_train` | `dataset/videos_train/` | Training set (ImageFolder: `class_0/1/2/`) |
| `videos_val` | `dataset/videos_val/` | Validation set (ImageFolder: `class_0/1/2/`) |

#### Checkpoints

| Save Path | Description |
|-----------|-------------|
| `ckpts/vjepa-nmode-pretrain.pt` | Pretrained ViT-Giant backbone (standard fine-tuning start point) |
| `ckpts/vjepa-nmode-pretrain-vitl.pt` | Pretrained ViT-Large backbone |
| `ckpts/vjepa-nmode-pretrain-vith.pt` | Pretrained ViT-Huge backbone |
| `ckpts/vjepa_full/best_vjepa_model9639(paper).pt` | Paper-reported fine-tuned ViT-Giant checkpoint |
| `ckpts/vjepa_full_vitl/best_vjepa_model.pt` | Fine-tuned ViT-Large checkpoint |
| `ckpts/vjepa_full_vith/best_vjepa_model.pt` | Fine-tuned ViT-Huge checkpoint |
| `distill/cls/output/ckpts/best_distill.pt` | Classification distillation — feature alignment (Stage 1) |
| `distill/cls/output/ckpts/best_finetune_distilled.pt` | Classification distillation — fine-tuned from distilled (Stage 2) |
| `distill/cls/output/ckpts/best_finetune_scratch.pt` | Classification distillation — fine-tuned from scratch baseline |
| `distill/det/output/ckpts/distilled.pt` | Detection distillation — YOLO feature alignment (Stage 1) |
| `distill/det/output/ckpts/ft-scratch.pt` | Detection distillation — YOLO fine-tuned from scratch baseline |

### 3. Prepare your customized dataset

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


</details>

## Project Structure
<details>
<summary>Show/Hide</summary>

```
BUVFM/
├── download_dataset_ckpt.py    # Download all datasets & model weights from Hugging Face Hub
├── train.py                  # Fine-tuning script
├── inference_ddp_old.py      # Multi-GPU distributed inference 
├── test.py                   # Multi-GPU distributed inference (recursive dir, no labels)
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
│   ├── prepare.py              # Auto-generate CSVs, YAML configs, and shell scripts
│   ├── app/                    # Pretraining entry points & VJEPA trainer
│   │   ├── main.py             # Single-GPU pretraining entry
│   │   ├── main_distributed.py # Multi-GPU distributed pretraining entry
│   │   ├── scaffold.py         # Config loading & trainer construction
│   │   ├── vjepa/              # VJEPA pretraining engine
│   │   └── vjepa_droid/        # VJEPA-DROID pretraining engine
│   ├── evals/                  # Fine-tuning & evaluation entry points
│   │   ├── main.py             # Single-GPU fine-tuning entry
│   │   ├── main_distributed.py # Multi-GPU distributed fine-tuning entry
│   │   ├── scaffold.py         # Eval config loading
│   │   └── video_classification_frozen/  # Frozen backbone evaluation
│   ├── configs/                # YAML templates & per-model configs
│   │   ├── pretrain-tample.yaml  # Pretrain config template
│   │   ├── fintune-tample.yaml   # Fine-tune config template
│   │   ├── vitg/               # Generated configs for ViT-Giant
│   │   ├── vith/               # Generated configs for ViT-Huge
│   │   └── vitl/               # Generated configs for ViT-Large
│   ├── src/                    # Shared models, datasets, masks, utils
│   └── tests/                  # Unit tests for models and datasets
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
├── ckpts/
├── assets/                         # 16 example videos (4 per category: Class0/1/2/NO)
└── output/                         # Inference result CSVs & t-SNE outputs

```

</details>

## Dataset Preparation
<details>
<summary>Show/Hide</summary>


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

</details>

## Checkpoint Preparation
<details>
<summary>Show/Hide</summary>

### Download Datasets & Checkpoints from Hugging Face Hub

Use `download_dataset_ckpt.py` to download all demo datasets and model weights (pretrained checkpoint, fine-tuned checkpoints, distillation checkpoints, and QC models) from [xenosscu/BUVFM](https://huggingface.co/xenosscu/BUVFM) on Hugging Face Hub:

```bash
HF_TOKEN=/your/hugging/face/token python download_dataset_ckpt.py
```

This downloads datasets to `./dataset/` and model weights to `./`, preserving the expected directory structure (`ckpts/`, `distill/`, `QC/`).

You can also specify custom output directories:

```bash
HF_TOKEN=... python download_dataset_ckpt.py ./my_dataset ./my_models
```

### Pretrained Starting Point

Fine-tuning starts from `./ckpts/vjepa-nmode-pretrain.pt`. Alternative pretrained checkpoints are available under `ckpts/` (e.g., MAE-pretrained `vitg_mae_e1.pt`, VAE-pretrained `vitg_vae_e1.pt`, ViT-Large `vjepa-vitl.pt`). See the [Pretraining](#pretraining) section for how these are produced.

### Fine-tuned Checkpoints

Fine-tuned checkpoints are saved under `ckpts/vjepa_full/` (or variant directories like `vjepa_full_maee1`, `vjepa_full_newval`, etc.). Naming convention: `best_vjepa_model.pt` (best by validation accuracy).

</details>

## Pretraining
<details>
<summary>Show/Hide</summary>

See `pretraining/README.md` .

</details>

## Training
<details>
<summary>Show/Hide</summary>

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

</details>

## Knowledge Distillation for Edge-Side Quality Control Model
<details>
<summary>Show/Hide</summary>

See [distill/README.md](distill/README.md) for details.

</details>

## Inference & Evaluation
<details>
<summary>Show/Hide</summary>

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
Recursively scans all videos in the given directory (no `class_N` folder structure required). Output CSV contains `video_name`, `p0`, `p1`, `p2` (no `label` column).

```bash
export TRPORT=29599
export CKPT=/path/to/ckpt
export VAL_DIR='/path/to/val/dir'
export OUT='test'
export EXTRA=""
export FS=2
export FPC=16
torchrun --master_port=$TRPORT --nproc_per_node=4 test.py \
    --val_dir $VAL_DIR --checkpoint $CKPT --output ./output/$OUT.csv \
    $EXTRA --frame_step $FS --frames_per_clip $FPC
```

</details>

## Demo
<details>
<summary>Show/Hide</summary>

### 2-Stage Pipeline Demo

A web-based interactive demo implementing a cascaded 2-stage inference pipeline:

```
Video Input → [Stage 1: Screening] → [Stage 2: Risk Grading]
                    │                         │
          MobileNetV3 + YOLO           VJEPA2 ViT-Giant
          No lesions? → STOP           3-class probs + Grad-CAM
```

Pipeline logic and model interfaces (MobileNetV3, YOLO, VJEPA2, Grad-CAM) are in `pipeline_utils.py`.

#### Running Locally

```bash
conda activate buvfm
python app.py
```

Open http://localhost:9530 in your browser. Upload a video or click an example from `assets/` (16 videos across 4 categories: Class 0 / Class NO / Class 1 / Class 2). The model is loaded on the first pipeline run.

Optional environment variables:

```bash
export BUVFM_CHECKPOINT_PATH=/absolute/path/to/checkpoint.pt
export BUVFM_ASSETS_DIR=/absolute/path/to/assets
export BUVFM_MAX_FILE_SIZE_MB=100
```

#### Expected Output

The demo displays results in a Gradio web UI with two stages:

1. **Stage 1 (Screening)**: Frame-level lesion detection with YOLO bounding boxes and MobileNetV3 classification. Videos with no detected lesions are flagged and the pipeline stops.
2. **Stage 2 (Risk Grading)**: 3-class BI-RADS risk probabilities (Class 0 / Class 1 / Class 2) with Grad-CAM heatmap overlays highlighting diagnostically relevant regions.

Example inference results are saved to `./output/` as CSV files with per-video predictions and confidence scores.

#### Expected Run Time

| Scenario | Time |
|----------|------|
| Model loading (first run, cold start) | ~10–30 seconds |
| Per-video inference (2-stage pipeline) | ~5–15 seconds |
| Total (load + 1 example video) | ~30–60 seconds |

*Times measured on a desktop with NVIDIA RTX 4090 (24 GB VRAM). On CPU-only machines the demo will run significantly slower and is not recommended for production use.*

### Online Demo

A hosted Gradio demo is available at [http://buvfm.machineilab.org/](http://buvfm.machineilab.org/).


</details>

## System Requirements
<details>
<summary>Show/Hide</summary>

### Operating System
| Item | Version |
|------|---------|
| OS | Ubuntu 22.04.5 LTS (Jammy Jellyfish) |
| Kernel | 6.14.0-37-generic |
| Architecture | x86_64 |

### Hardware

#### Minimum
| Component | Specification |
|-----------|---------------|
| CPU | 8-core x86_64 processor |
| RAM | 32 GB |
| GPU | 1× NVIDIA GPU with 24 GB VRAM (e.g., RTX 4090) |

#### Recommended (this configuration)
| Component | Specification |
|-----------|---------------|
| CPU | Intel Xeon Platinum 8568Y+ (192 cores) |
| RAM | 125 GB |
| GPU | 4× NVIDIA RTX PRO 6000 Blackwell Server Edition (96 GB VRAM each) |

### GPU & CUDA
| Item | Version |
|------|---------|
| GPU Driver | 570.153.02 |
| CUDA | 12.8 |
| cuDNN | Compatible with CUDA 12.x |

### Software Dependencies
| Package | Version |
|---------|---------|
| Python | 3.10.9 |

For all Python package dependencies, see [requirements.txt](requirements.txt).

### Tested Versions
| Item | Tested Version |
|------|----------------|
| Ubuntu | 22.04.5 LTS |
| Linux Kernel | 6.14.0-37-generic |
| Python | 3.10.9 |
| CUDA | 12.8 |
| GPU Driver | 570.153.02 |
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition (96 GB) |
| PyTorch | ≥ 2.0.0 |

</details>

## Reproduction
<details>
<summary>Show/Hide</summary>

To reproduce the quantitative results reported in the manuscript:

### 1. Prepare Environment & Data

```bash
conda create -n buvfm python=3.10.9 -y
conda activate buvfm
pip install -r requirements.txt
HF_TOKEN=/your/hugging/face/token python download_dataset_ckpt.py
```

Organize your dataset following the [Dataset Preparation](#dataset-preparation) structure.

### 2. Fine-Tune

```bash
torchrun --nproc_per_node=4 train.py \
    --pretrained_ckpt ./ckpts/vjepa-nmode-pretrain.pt \
    --train_dir /path/to/dataset/train \
    --val_dir /path/to/dataset/val \
    --batch_size 16 \
    --lr 5e-5 \
    --epochs 50
```

Checkpoints are saved to `./ckpts/vjepa_full/best_vjepa_model.pt`. Logs are written to `./output/logs_vjepa/vjepa_full.log`.

### 3. Evaluate

```bash
torchrun --master_port=29599 --nproc_per_node=4 inference_ddp_old.py \
    --val_dir /path/to/dataset/val \
    --checkpoint ./ckpts/vjepa_full/best_vjepa_model.pt \
    --output ./output/results.csv \
    --frame_step 2 --frames_per_clip 16
```

Per-video predictions and confidence scores are written to the output CSV.

### 4. Robustness Evaluation (Speckle Noise)

```bash
torchrun --master_port=29599 --nproc_per_node=4 inference_ddp_old.py \
    --val_dir /path/to/dataset/val \
    --checkpoint ./ckpts/vjepa_full/best_vjepa_model.pt \
    --output ./output/robustness.csv \
    --restore_true
```

This evaluates across speckle noise ratios from 0.05 to 0.95.

### 5. Visualization (t-SNE / PCA)

```bash
python cluster_videos_nocohen.py
```

Edit `checkpoint_path` and `val_dir` inside the script before running.

</details>

## License
<details>
<summary>Show/Hide</summary>

This model and associated code are released under the [CC-BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/) license and may only be used for non-commercial, academic research purposes with proper attribution. Any commercial use, sale, or other monetization of the BUVFM model and its derivatives, which include models trained on outputs from the BUVFM model or datasets created from the BUVFM model, is prohibited and requires prior approval.

</details>
