# VJEPA2 Video Classification

Fine-tuning [VJEPA2](https://github.com/facebookresearch/jepa) (Video Joint-Embedding Predictive Architecture) for medical ultrasound video classification. Built on the ViT-Giant backbone with an Attentive Classifier head, supporting multi-GPU distributed training with mixed precision.

## Model Architecture

- **Backbone**: ViT-Giant (`vit_giant_xformers`), 1408 embed dim, 40 layers, 224×224 input
- **Patch embedding**: 3D tubelets of 2×16×16 (temporal×spatial)
- **Clip aggregation**: Multi-clip temporal concatenation with optional positional embeddings
- **Classifier**: Attentive Classifier with cross-attention pooling (16 heads, 1 probe block)
- **Output**: 3-class classification (class_0, class_1, class_2)

## Project Structure

```
vjepa/
├── train.py              # Distributed training with DDP + AMP
├── inference.py          # Validation set inference & evaluation
├── test.py               # General-purpose inference on arbitrary video dirs
├── vis.py                # Grad-CAM visualization for model interpretability
├── app.py                # Gradio web application (3-stage pipeline demo)
├── pipeline_utils.py     # Pipeline logic, model interfaces, shared utilities
├── buildmodel.py         # Model construction & checkpoint loading
├── mydataset.py          # VideoFolderDataset (ImageFolder-style for videos)
├── utils.py              # Data augmentation & video transforms
├── src/
│   ├── models/
│   │   ├── vision_transformer.py   # ViT backbone
│   │   ├── attentive_pooler.py     # Attentive classifier head
│   │   └── ac_predictor.py         # Predictor module
│   ├── masks/                      # Masking utilities
│   ├── datasets/                   # Dataset utils & transforms
│   └── utils/                      # Distributed, logging, schedulers
├── ckpts/                          # Saved checkpoints (gitignored)
├── assets/                         # Example videos for demo (15 videos, 5 per class)
├── convert_video.py               # Transcode video to H.264 for browser playback
├── logs_vjepa/                     # Training logs (gitignored)
└── csvs/                           # Inference results (gitignored)
```

## Dependencies

- Python >= 3.8
- PyTorch >= 2.0 (CUDA)
- torchvision
- [decord](https://github.com/dmlc/decord) (video decoding)
- scikit-learn
- pandas
- tqdm
- numpy
- opencv-python
- gradio
- plotly
- imageio

Install:

```bash
pip install torch torchvision decord scikit-learn pandas tqdm numpy opencv-python gradio plotly
```

## Dataset Preparation

Organize videos in an `ImageFolder`-style directory structure — each subfolder named `class_N` holds videos of that class:

```
dataset/
├── train/
│   ├── class_0/
│   │   ├── video_001.mp4
│   │   ├── video_002.mp4
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

> **Note**: The training script applies undersampling to class_0 (capped at 10,000 samples by default) to handle class imbalance, and uses weighted cross-entropy loss.

## Checkpoint Preparation

### Pretrained VJEPA2 Checkpoint

Download the VJEPA2 pretrained weights from the official source:

- [VJEPA2 ViT-Giant checkpoint](https://dl.fbaipublicfiles.com/jepa/vjepa2/vit_giant.pth) or use your own pre-trained checkpoint.

Place the checkpoint file (e.g., `checkpoint_epoch_20.pt`) in the project root or a path of your choice.

### Fine-tuned Checkpoints

Fine-tuned checkpoints are saved under `ckpts/vjepa_full/` during training. The naming convention is `best_vjepa_model{accuracy}.pt`, e.g., `best_vjepa_model9720.pt` (97.20% val accuracy).

## Training

### Full Finetuning (encoder + classifier)

```bash
# Stage 1: from pretrained VJEPA2 weights
torchrun --nproc_per_node=2 train.py \
    --pretrained_ckpt ./checkpoint_epoch_20.pt \
    --batch_size 16 \
    --lr 5e-5 \
    --epochs 50

# Stage 2: continue from best Stage 1 checkpoint
torchrun --nproc_per_node=4 train.py \
    --pretrained_ckpt ./ckpts/vjepa_full/best_vjepa_model9422.pt \
    --batch_size 16 \
    --lr 5e-5 \
    --epochs 50

# Stage 3: further finetune with lower learning rate
torchrun --nproc_per_node=4 train.py \
    --pretrained_ckpt ./ckpts/vjepa_full/best_vjepa_model9647.pt \
    --batch_size 16 \
    --lr 1e-5 \
    --epochs 50

# Stage 4: fine-grained finetuning
torchrun --nproc_per_node=4 train.py \
    --pretrained_ckpt ./ckpts/vjepa_full/best_vjepa_model9720.pt \
    --batch_size 16 \
    --lr 5e-6 \
    --epochs 50
```

### Freeze Backbone (classifier only)

```bash
torchrun --nproc_per_node=4 train.py \
    --pretrained_ckpt ./checkpoint_epoch_20.pt \
    --batch_size 16 \
    --lr 5e-5 \
    --freeze_backbone
```

### Training Arguments

| Argument | Default | Description |
|---|---|---|
| `--pretrained_ckpt` | (required) | Path to checkpoint file |
| `--batch_size` | 4 | Batch size per GPU |
| `--lr` | 5e-5 | Learning rate |
| `--epochs` | 50 | Number of training epochs |
| `--num_frames` | 16 | Frames per clip |
| `--freeze_backbone` | False | Freeze encoder, train classifier only |

### Training Output

- **Logs**: `./logs_vjepa/vjepa_full.log` (or `vjepa_frozen.log` if `--freeze_backbone`)
- **Checkpoints**: `./ckpts/vjepa_full/best_vjepa_model.pt` (best by val accuracy)
- **Eval CSV**: `./ckpts/vjepa_full/best_vjepa_eval.csv` (per-video predictions)

## Inference & Evaluation

### Validation set evaluation (with labels)

```bash
python inference.py
```

Edit the following variables in `inference.py` before running:
- `checkpoint_path` — path to the fine-tuned checkpoint
- `val_dir` — path to the validation dataset (same structure as training)
- `output_csv` — output CSV path (default: `./csvs/inference_results.csv`)

Output includes per-video probabilities (`p0`, `p1`, `p2`), overall accuracy, and confusion matrix.

### Inference on arbitrary video directory (no labels required)

```bash
python test.py
```

Edit the following variables in `test.py` before running:
- `checkpoint_path` — path to the fine-tuned checkpoint
- `input_dir` — path to directory containing videos (recursive scan)
- `output_csv` — output CSV path (default: `./csvs/test_results.csv`)

Supports mixed precision inference and outputs per-video class probabilities.

## Visualization

### Grad-CAM Heatmap Visualization

```bash
python vis.py
```

This generates Grad-CAM heatmaps overlaid on video frames to interpret which spatiotemporal regions the model attends to for classification. Edit the following parameters in `visualize_gradcam()` before running:

| Parameter | Default | Description |
|---|---|---|
| `video_path` | `./assets/CLASS1_0.mp4` | Input video path |
| `checkpoint_path` | `./ckpts/vjepa_full/best_vjepa_model9720.pt` | Model checkpoint |
| `output_dir` | `./output` | Output directory |
| `target_class` | None | Grad-CAM target class (None = use predicted class) |
| `alpha` | 0.5 | Heatmap overlay transparency |

Output files (under `output_dir`):
- `{video_name}_gradcam.mp4` — heatmap overlay video
- `{video_name}_gradcam_compare.mp4` — side-by-side comparison (original | Grad-CAM)
- `{video_name}_gradcam_summary.png` — summary image with top row: original frames, bottom row: heatmaps

## Video Conversion

Source videos encoded with mpeg4 or other codecs may not play in browsers (e.g., Gradio's `gr.Video` component). Use the conversion script to transcode them to H.264:

```bash
python convert_video.py input.mp4 output.mp4
python convert_video.py input.mp4 output.mp4 --fps 8   # specify output FPS
```

This uses `imageio` with the `libx264` encoder to produce browser-compatible mp4 files. The `--fps` flag is optional; if omitted, the source FPS is preserved.

## 3-Stage Pipeline Demo (Gradio)

A web-based interactive demo implementing a cascaded 3-stage inference pipeline for medical ultrasound video classification:

```
Video Input → [Stage 1: Rapid Filter] → [Stage 2: Refinement] → [Stage 3: VJEPA2 Classify]
                    │                           │                           │
              No lesions?                  No lesions?               3-class probs
               → STOP                      → STOP                 + Grad-CAM heatmap
```

### Stage 1 — Rapid Negative Filtering

Lightweight classifier (MobileNet) quickly rejects clearly negative videos. Only videos classified as containing lesions proceed to a lightweight detector (YOLO) for bounding box localization. If no boxes are found, the pipeline stops.

- **Classifier**: MobileNet-V3 (placeholder — random 70% positive rate)
- **Detector**: YOLOv8-nano (placeholder — random 80% box-found rate)

### Stage 2 — Medium-weight Refinement

nmODE-ResNet provides a second filter to further reduce false positives before expensive VJEPA2 inference.

- **Model**: nmODE-ResNet18 (placeholder — random 80% positive rate)

### Stage 3 — VJEPA2 Classification

Full VJEPA2 ViT-Giant model performs 3-class risk classification (low / medium / high) with Grad-CAM visualization for interpretability.

- **Model**: VJEPA2 ViT-Giant (real, fine-tuned)
- **Output**: 3-class probability bar chart + Grad-CAM heatmap video + summary image

### Running the Demo

```bash
pip install gradio plotly
python app.py
```

Open http://localhost:9530 in your browser. Upload a video or click an example to run the pipeline. The flowchart at the top updates dynamically as each stage completes.

> **Note**: Stage 1 and 2 currently use placeholder models (random outputs) for demonstration. Replace the `classify()` / `detect_boxes()` methods in `pipeline_utils.py` with real model inference when available.

## Results

| Checkpoint | Val Accuracy |
|---|---|
| best_vjepa_model9422.pt | 94.22% |
| best_vjepa_model9526.pt | 95.26% |
| best_vjepa_model9639.pt | 96.39% |
| best_vjepa_model9647.pt | 96.47% |
| best_vjepa_model9663.pt | 96.63% |
| best_vjepa_model9720.pt | 97.20% |

## License

This project is based on [VJEPA](https://github.com/facebookresearch/jepa) by Meta Platforms, Inc., licensed under the MIT License.
