```bash
# MAE pretraining
python pretraining/prepare_data.py --video_dirs /path/to/videos --output_dir pretraining/data
torchrun --nproc_per_node=8 pretraining/run_pretrain_mae.py \
    --data_root /path/to/dataset --data_path pretraining/data/us_videomae_train.txt \
    --output_dir pretraining/output/mae_vitg --batch_size 16 --epochs 100
python pretraining/convert_checkpoint.py --method videomae \
    --input pretraining/output/mae_vitg/checkpoint-99.pth --output ./ckpts/vitg_mae.pt

# VAE pretraining
torchrun --nproc_per_node=8 pretraining/run_pretrain_vae.py \
    --data_root /path/to/dataset --data_path pretraining/data/us_videomae_train.txt \
    --output_dir pretraining/output/vae --batch_size 2 --epochs 100
python pretraining/convert_checkpoint.py --method vae \
    --input pretraining/output/vae/checkpoint-99.pth --output ./ckpts/vitg_vae.pt
```



# VJEPA2 VideoMAEv2 Pretraining

基于 VideoMAEv2 的 MAE (Masked Autoencoder) 自监督预训练方案，在 500K+ 无标签超声视频上预训练 ViT-g 模型。


## 目录结构

```
pretraining/
├── run_pretrain_mae.py              # MAE 预训练入口脚本（独立）
├── run_pretrain.py                  # 统一启动器（VideoMAEv2 + BEVT）
├── convert_checkpoint.py            # 预训练 ckpt → buildmodel 格式转换
├── prepare_data.py                  # 生成标注文件（mp4 目录 → .txt）
├── methods/                         # 提取的预训练方法（自包含）
│   ├── __init__.py
│   ├── mae_masking.py               # TubeMaskingGenerator, RunningCellMaskingGenerator
│   ├── mae_utils.py                 # cosine_scheduler, NativeScaler, DDP, MetricLogger
│   ├── mae_dataset.py               # VideoMAEPretrainDataset, DataAugmentationForVideoMAE
│   ├── mae_modeling_pretrain.py     # VJEPAEncoder, MAEDecoder, MAEPretrainModel
│   └── mae_engine.py               # train_one_epoch (per-patch normalized MSE)
├── output/                          # 输出目录（日志、checkpoint）
├── data/                            # 标注文件（生成的 us_videomae_train.txt）
└── VideoMAEv2/                      # 原始 VideoMAEv2 参考实现
```

## 使用方法

### 1. 准备数据

```bash
python pretraining/prepare_data.py \
    --video_dirs /home/wcz/workspace/DATASET/us_foundation_model_dataset_videos_videos \
                 /home/wcz/workspace/DATASET/us_foundation_model_dataset_img_videos \
    --data_root /home/wcz/workspace/DATASET \
    --output_dir pretraining/data
```

生成 `pretraining/data/us_videomae_train.txt`，每行格式: `video_path 0 -1`

### 2. 启动预训练

```bash
# 4 GPU
torchrun --nproc_per_node=4 pretraining/run_pretrain_mae.py \
    --data_root /home/wcz/workspace/DATASET \
    --data_path pretraining/data/us_videomae_train.txt \
    --output_dir pretraining/output/mae_vitg \
    --batch_size 16 --epochs 5

# 单 GPU (调试)
torchrun --nproc_per_node=1 pretraining/run_pretrain_mae.py \
    --data_root /home/wcz/workspace/DATASET \
    --data_path pretraining/data/us_videomae_train.txt \
    --output_dir pretraining/output/mae_vitg_debug \
    --batch_size 2 --num_sample 1 --epochs 5
```

### 3. 转换 checkpoint 供 downstream 使用

```bash
python pretraining/convert_checkpoint.py \
    --method videomae \
    --input pretraining/output/mae_vitg/checkpoint-299.pth \
    --output pretraining/output/mae_vitg/encoder_checkpoint.pt
```

### 4. 在下游任务中加载

```python
from buildmodel import build_model

model, classifier = build_model(
    checkpoint_path="pretraining/output/mae_vitg/encoder_checkpoint.pt",
    resolution=224,
    frames_per_clip=16,
    num_classes=3,
    num_heads=16,
    num_probe_blocks=1,
)
```


## VideoVAEPlus 预训练

基于 VideoVAEPlus 的视频 VAE 自编码器预训练，在无标签超声视频上学习视频压缩表示。

### 目录结构

```
pretraining/
├── run_pretrain_vae.py                # VAE 预训练入口脚本
├── methods/
│   ├── vae_modeling_pretrain.py       # Encoder2plus1D, Decoder2plus1D, TemporalCNN, LPIPS+3D GAN loss
│   ├── vae_dataset.py                 # VideoVAEPretrainDataset (同 MAE 的视频加载，无掩码)
│   ├── vae_engine.py                  # train_one_epoch (双优化器: AE + Discriminator)
│   └── vae_utils.py                   # 复用 mae_utils 的工具函数
└── VideoVAEPlus/                      # 原始 VideoVAEPlus 参考实现
```

### 使用方法

```bash
# 1. 训练 (8 GPUs)
torchrun --nproc_per_node=8 pretraining/run_pretrain_vae.py --data_root /home/lx/alg/videos_val/class_2   --data_path pretraining/data/us_videomae_train.txt --output_dir pretraining/output/vae_4z --batch_size 2 --epochs 10

# 2. 转换检查点
python pretraining/convert_checkpoint.py --method vae \
    --input pretraining/output/vae_4z/checkpoint-99.pth \
    --output pretraining/output/vae_4z/encoder_checkpoint.pt

# 3. 在下游任务中加载 (自动检测)
from buildmodel import build_model
model, classifier = build_model(
    checkpoint_path="pretraining/output/vae_4z/encoder_checkpoint.pt",
    resolution=224, frames_per_clip=16, num_classes=3
)
```

## 参考

- VideoMAEv2: https://github.com/OpenGVLab/VideoMAEv2
- VideoVAEPlus: `pretraining/VideoVAEPlus/`
- VJEPA: `src/models/vision_transformer.py`
- 下游模型构建: `buildmodel.py`
