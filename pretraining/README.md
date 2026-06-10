# BUVFM — Pretraining

BUVFM is forked and modified from [V-JEPA2](https://github.com/facebookresearch/vjepa2).
This project focuses on introducing an nmODE-enhanced predictor into the V-JEPA2 pretraining framework and provides a full fine-tuning pipeline for downstream video classification tasks.

## Overview

The main goal of this repository is to explore nmODE-based temporal representation modeling within the V-JEPA2 self-supervised video pretraining framework.

Compared with the original V-JEPA2 implementation, this project mainly modifies the pretraining predictor architecture and provides downstream full fine-tuning support for video datasets.

## Main Modifications

The key modifications are:

* `src/models/nmODE.py`
  Implements the nmODE module used in the modified predictor.

* `src/models/predictor.py`
  Integrates nmODE into the V-JEPA2 pretraining predictor architecture.

* `evals/video_classification_frozen/eval.py`
  Provides the downstream full fine-tuning pipeline using pretrained weights on video classification datasets.

* `configs/pretrain-tample.yaml`
  Pretraining configuration template for the nmODE-enhanced V-JEPA2 model.

* `configs/fintune-tample.yaml`
  Fine-tuning configuration template for video classification experiments.

## Repository Structure

```text
BUVFM/
├── app/                         # Pretraining entry points and training utilities
├── configs/                     # Pretraining, evaluation, and inference configs
├── evals/                       # Downstream evaluation and fine-tuning code
├── src/
│   ├── datasets/                # Dataset loading and video transforms
│   ├── masks/                   # Mask generation modules
│   ├── models/                  # V-JEPA2 models, predictor, and nmODE module
│   └── utils/                   # Distributed training, logging, checkpoints, schedulers
├── tests/                       # Unit tests
├── requirements.txt             # Python dependencies
├── setup.py                     # Package setup
└── prepare.py                   # Auto-generate CSVs, YAML configs, and shell scripts
```

## Installation

Follow the instructions of [V-JEPA2](https://github.com/facebookresearch/vjepa2).

## Checkpoints, Dataset and Config Preparation

### 1. Download Pretrained Weights

Download the V-JEPA2 pretrained weights from [V-JEPA2](https://github.com/facebookresearch/vjepa2) and place them under `configs/{model}/`:

```bash
# ViT-Giant (~1.1B params)
wget https://dl.fbaipublicfiles.com/vjepa2/vjepa2-vitg16-256-c4800.pt -O configs/vitg/vitg.pt

# ViT-Huge (~632M params)
wget https://dl.fbaipublicfiles.com/vjepa2/vjepa2-vith16-256-c4800.pt -O configs/vith/vith.pt

# ViT-Large (~307M params)
wget https://dl.fbaipublicfiles.com/vjepa2/vjepa2-vitl16-256-c4800.pt -O configs/vitl/vitl.pt
```

### 2. Prepare Datasets

Organize your datasets as follows:

**Pretrain dataset** — self-supervised pretraining (flat directory or nested subdirectories, any structure):

```
/path/to/pretrain/
├── video1.mp4
├── subdir/
│   ├── video2.mp4
│   └── video3.mp4
└── ...
```

**Fine-tune dataset** — supervised classification (ImageFolder structure, `class_N` subfolders):

```
/path/to/finetune/
├── train/
│   ├── class_0/
│   ├── class_1/
│   └── class_2/
└── val/
    ├── class_0/
    ├── class_1/
    └── class_2/
```

### 3. Generate Configs with `prepare.py`

`prepare.py` automatically generates CSV manifests, YAML configs, and launch scripts:

```bash
python prepare.py --model vitg \
    --pretrain_dataset /path/to/pretrain \
    --train_dataset /path/to/finetune/train \
    --val_dataset /path/to/finetune/val
```

**Model options:**

| `--model` | Backbone | `out_layers` | Download URL |
|-----------|----------|--------------|--------------|
| `vitg` | `vit_giant_xformers` | [39] | https://dl.fbaipublicfiles.com/vjepa2/vjepa2-vitg16-256-c4800.pt |
| `vith` | `vit_huge` | [31] | https://dl.fbaipublicfiles.com/vjepa2/vjepa2-vith16-256-c4800.pt |
| `vitl` | `vit_large` | [23] | https://dl.fbaipublicfiles.com/vjepa2/vjepa2-vitl16-256-c4800.pt |

**Optional arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--pretrain_csv` | `pretrain.csv` | Pretrain CSV filename |
| `--train_csv` | `train.csv` | Train CSV filename |
| `--val_csv` | `val.csv` | Validation CSV filename |
| `--devices` | `cuda:0` | Device(s) for shell scripts |
| `--pretrain_batch_size` | 32 | Pretrain batch size (used for ipe calculation) |
| `--finetune_batch_size` | 16 | Fine-tune batch size |

**Output structure:**

```
configs/{model}/
├── pretrain.csv        # Video manifest for pretraining
├── train.csv           # Video manifest for fine-tuning (train)
├── val.csv             # Video manifest for fine-tuning (val)
├── pretrain.yaml       # Pretrain config (checkpoint path auto-filled)
├── finetune.yaml       # Fine-tune config (references pretrain output)
├── pretrain.sh         # Launch pretraining
├── finetune.sh         # Launch fine-tuning (runs after pretraining)
└── {model}.pt          # Downloaded pretrained weights (place here)
```

After running `prepare.py`, download the weights into `configs/{model}/{model}.pt` before launching training.

## Pretraining

### Self-Supervised Pretraining (V-JEPA2 with nmODE)

```bash
python -m app.main --fname configs/vitg/pretrain.yaml --devices cuda:0
```

Multi-GPU distributed pretraining:

```bash
torchrun --nproc_per_node=4 --master_port=29501 -m app.main_distributed \
    --fname configs/vitg/pretrain.yaml --devices cuda:0 cuda:1 cuda:2 cuda:3
```

### Fine-Tuning (Video Classification)

```bash
PYTHONPATH=. python -m evals.main --fname configs/vitg/finetune.yaml --devices cuda:0
```

Multi-GPU distributed fine-tuning:

```bash
torchrun --nproc_per_node=4 --master_port=29501 -m evals.main_distributed \
    --fname configs/vitg/finetune.yaml --devices cuda:0 cuda:1 cuda:2 cuda:3
```

Or simply use the generated shell scripts:

```bash
bash configs/vitg/pretrain.sh
bash configs/vitg/finetune.sh
```

## Key Config Parameters

### Pretrain

| Parameter | Value | Description |
|-----------|-------|-------------|
| `model_name` | `vit_giant_xformers` | ViT backbone variant |
| `predictor.out_layers` | [39] | Predictor output layers |
| `predictor.use_nmODE` | true | Enable nmODE-enhanced predictor |
| `optimization.ipe` | 500 | Iterations per epoch (auto-calculated by prepare.py) |
| `optimization.batch_size` | 32 | Batch size per GPU |
| `data.num_frames` | 16 | Number of frames per clip |
| `data.pin_mem` | true | Pin memory for faster data transfer |

### Fine-Tune

| Parameter | Value | Description |
|-----------|-------|-------------|
| `model_name` | `vit_giant_xformers` | ViT backbone variant |
| `classifier` | `mlp` | Classifier head type |
| `optimization.ipe` | 100 | Iterations per epoch |
| `optimization.batch_size` | 16 | Batch size per GPU |
| `data.num_frames` | 16 | Number of frames per clip |
| `evaluation.eval_every_iterations` | 100 | Evaluation frequency |

## Advanced Configuration

### Modify Model Architecture

Edit the following in the YAML config files:

- `model_name`: Choose from `vit_giant_xformers`, `vit_large`, `vit_huge`, `vit_small`, `vit_tiny`
- `predictor.out_layers`: Specify which transformer layers to use (e.g., [23] for ViT-Large)
- `predictor.use_nmODE`: Enable/disable nmODE module (default: true)

### Adjust Training Parameters

- `optimization.ipe`: Iterations per epoch (auto-calculated by `prepare.py` as dataset_size / batch_size)
- `optimization.batch_size`: Batch size per GPU
- `optimization.lr`: Learning rate
- `optimization.warmup`: Warmup iterations
- `data.num_frames`: Number of video frames per clip
- `data.spatial_size`: Input spatial resolution (default: 256)

### Data Augmentation

Configure augmentation in the YAML config:

```yaml
data:
  color_jitter: 0.0          # Color jitter strength (0 to disable)
  random_resize_scale: [0.3, 1.0]  # Random crop scale range
  reprob: 0.0                # Random erasing probability
  use_random_erasing: false  # Enable random erasing augmentation
```

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce `batch_size` in the YAML config
2. **Missing checkpoint**: Ensure pretrained weights are downloaded to the correct path
3. **Data loading errors**: Verify CSV file paths and video file formats
4. **Slow training**: Enable `data.pin_mem: true` and increase `data.num_workers`

### Performance Tips

- Use `torchrun` for multi-GPU training to significantly reduce training time
- Enable `data.pin_mem: true` for faster data transfer to GPU
- Adjust `data.num_workers` based on CPU cores (typically 4-8 per GPU)
- For large datasets, increase `optimization.ipe` to process more data per epoch

## Citation

If you use this code in your research, please cite:

```bibtex
@article{buvfm2025,
  title={BUVFM: Breast Ultrasound Video Foundation Model},
  author={},
  journal={arXiv preprint},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.

## Acknowledgments

- [V-JEPA2](https://github.com/facebookresearch/vjepa2) for the base pretraining framework
- [nmODE](https://github.com/2512988161/nmODE-V-pp) for the neural ODE module
