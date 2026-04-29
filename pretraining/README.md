# VJEPA2 VideoMAEv2 Pretraining

基于 VideoMAEv2 的 MAE (Masked Autoencoder) 自监督预训练方案，在 500K+ 无标签超声视频上预训练 ViT-g 模型。

## 目录结构

```
pretrain/
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

## 架构

### Encoder: VJEPA `vit_giant_xformers`
- 1408-dim, 40 layers, 22 heads, RoPE + SwiGLU FFN (mlp_ratio=48/11)
- `PatchEmbed3D` (tubelet=2, patch=16) → 1568 tokens from [16, 224, 224]
- 无位置编码（RoPE 内置于 attention）
- `VJEPAEncoder` 适配层：布尔 mask `[B, N]` → 索引 mask → `VisionTransformer.forward(x, masks=[ids_keep])`
- Gradient checkpointing 应用于所有 block

### Decoder: 浅层 Transformer
- 512-dim, 4 layers, 8 heads, standard attention (无 RoPE), GELU MLP
- 使用 VJEPA 的 `Block` (`src/models/utils/modules.py`) with `use_rope=False`
- 可学习的 mask token + sincos 位置编码
- 输出 head: 512 → 1536 (3 × 2 × 16² pixels per tubelet-patch)

### Masking 策略（VideoMAEv2 核心创新）
- **Encoder**: Tube masking (90%) — 同一随机空间 mask 应用于所有帧，形成时空"管道"
- **Decoder**: Running-cell masking (50%) — 2×2 cell 模式逐帧滑动，增加重建难度
- **Loss**: Per-patch normalized MSE，仅在 encoder-masked AND decoder-visible 的位置计算

### Checkpoint 兼容性
- 模型 state dict key 格式: `encoder.vit.patch_embed.proj.weight`, `encoder.vit.blocks.0.norm1.weight`
- `convert_checkpoint.py` 自动剥离 `encoder.vit.` 前缀 → `{"encoder": {...}, "classifiers": []}`
- 可直接通过 `buildmodel.build_model()` 加载

## 使用方法

### 1. 准备数据

```bash
python pretrain/prepare_data.py \
    --video_dirs /home/wcz/workspace/DATASET/us_foundation_model_dataset_videos_videos \
                 /home/wcz/workspace/DATASET/us_foundation_model_dataset_img_videos \
    --data_root /home/wcz/workspace/DATASET \
    --output_dir pretrain/data
```

生成 `pretrain/data/us_videomae_train.txt`，每行格式: `video_path 0 -1`

### 2. 启动预训练

```bash
# 4 GPU
torchrun --nproc_per_node=4 pretrain/run_pretrain_mae.py \
    --data_root /home/wcz/workspace/DATASET \
    --data_path pretrain/data/us_videomae_train.txt \
    --output_dir pretrain/output/mae_vitg \
    --batch_size 16 --epochs 5

# 单 GPU (调试)
torchrun --nproc_per_node=1 pretrain/run_pretrain_mae.py \
    --data_root /home/wcz/workspace/DATASET \
    --data_path pretrain/data/us_videomae_train.txt \
    --output_dir pretrain/output/mae_vitg_debug \
    --batch_size 2 --num_sample 1 --epochs 5
```

### 3. 转换 checkpoint 供 downstream 使用

```bash
python pretrain/convert_checkpoint.py \
    --method videomae \
    --input pretrain/output/mae_vitg/checkpoint-299.pth \
    --output pretrain/output/mae_vitg/encoder_checkpoint.pt
```

### 4. 在下游任务中加载

```python
from buildmodel import build_model

model, classifier = build_model(
    checkpoint_path="pretrain/output/mae_vitg/encoder_checkpoint.pt",
    resolution=224,
    frames_per_clip=16,
    num_classes=3,
    num_heads=16,
    num_probe_blocks=1,
)
```

## 关键参数 (ViT-g)

| 参数 | 值 | 说明 |
|------|------|------|
| batch_size | 4 (per GPU) | |
| epochs | 300 | |
| lr | 6e-4 | 线性缩放前 |
| warmup_epochs | 30 | |
| min_lr | 1e-5 | |
| warmup_lr | 1e-6 | |
| opt | adamw | β=(0.9, 0.95), eps=1e-8 |
| weight_decay | 0.05 | |
| clip_grad | 0.02 | Giant 专用 |
| mask_ratio | 0.90 | Encoder tube masking |
| decoder_mask_ratio | 0.50 | Decoder running-cell masking |
| decoder_depth | 4 | |
| num_frames | 16 | 每 clip 帧数 |
| sampling_rate | 4 | 帧间隔 (跨 64 帧) |
| num_sample | 4 | 重复增强样本数 |
| input_size | 224 | |
| tubelet_size | 2 | |
| with_checkpoint | True | Gradient checkpointing |
| normlize_target | True | Per-patch normalization |
| LR 缩放 | lr × total_batch / 256 | |

## 验证结果

| 测试 | 结果 |
|------|------|
| Mask 生成器 | `(8, 196)`, masked ratios: 0.898 / 0.500 |
| 模型前向 | 1026.3M params, output `[2, 784, 1536]` |
| Encoder state dict | 484 `encoder.vit.*` keys |
| build_model 加载 | "All keys matched successfully" |
| 端到端推理 | Encoder `[1, 1568, 1408]` → Classifier `[1, 3]` |

## VideoVAEPlus 预训练

基于 VideoVAEPlus 的视频 VAE 自编码器预训练，在无标签超声视频上学习视频压缩表示。

### 目录结构

```
pretrain/
├── run_pretrain_vae.py                # VAE 预训练入口脚本
├── methods/
│   ├── vae_modeling_pretrain.py       # Encoder2plus1D, Decoder2plus1D, TemporalCNN, LPIPS+3D GAN loss
│   ├── vae_dataset.py                 # VideoVAEPretrainDataset (同 MAE 的视频加载，无掩码)
│   ├── vae_engine.py                  # train_one_epoch (双优化器: AE + Discriminator)
│   └── vae_utils.py                   # 复用 mae_utils 的工具函数
└── VideoVAEPlus/                      # 原始 VideoVAEPlus 参考实现
```

### 架构

**Encoder:** 2+1D 卷积网络 (Encoder2plus1D)
- ch=128, ch_mult=[1,2,4,4], 2 个 ResNet block，无 spatial attention
- 输入 [B,3,T,H,W] → [B, 2*z_channels, T, H/16, W/16]
- 中间层: AttnBlock3D + TemporalAttention (相对位置编码)

**Temporal Compression:** 1D CNN (EncoderTemporal1DCNN / DecoderTemporal1DCNN)
- temporal_scale_factor=4: 将 T 帧压缩为 T/4 个 latent 帧
- hidden_channel=128, 通过 stride-2 3D convs 实现 2 级下采样

**Decoder:** 2+1D 卷积网络 (Decoder2plus1D)
- 对称上采样 + TemporalAttention

**Loss:** L1 + LPIPS + KL + 3D PatchGAN 判别器
- `disc_start=50001`: 前 50k 步仅使用重建损失
- `kl_weight=1e-6`, `disc_weight=0.5`

### Checkpoint 兼容性

- `convert_checkpoint.py --method vae` 提取 `encoder.*`, `quant_conv.*`, `encoder_temporal.*` 键
- 输出 `{"encoder": {...}, "classifiers": []}` 格式
- `buildmodel.py` 自动检测 VAE 检查点并构建 VAEEncoderWrapper

### 使用方法

```bash
# 1. 训练 (8 GPUs)
torchrun --nproc_per_node=8 pretrain/run_pretrain_vae.py --data_root /home/lx/alg/videos_val/class_2   --data_path pretrain/data/us_videomae_train.txt --output_dir pretrain/output/vae_4z --batch_size 2 --epochs 10

# 2. 转换检查点
python pretrain/convert_checkpoint.py --method vae \
    --input pretrain/output/vae_4z/checkpoint-99.pth \
    --output pretrain/output/vae_4z/encoder_checkpoint.pt

# 3. 在下游任务中加载 (自动检测)
from buildmodel import build_model
model, classifier = build_model(
    checkpoint_path="pretrain/output/vae_4z/encoder_checkpoint.pt",
    resolution=224, frames_per_clip=16, num_classes=3
)
```

### 关键参数 (VAE 4z)

| 参数 | 值 | 说明 |
|------|------|------|
| batch_size | 2 (per GPU) | |
| epochs | 100 | |
| lr | 4.5e-6 | 线性缩放前 |
| warmup_epochs | 5 | |
| opt | adamw | β=(0.5, 0.9), eps=1e-8 |
| clip_grad | 0.02 | |
| embed_dim | 4 | VAE latent 维度 |
| z_channels | 4 | 编码器中间通道数 |
| temporal_scale_factor | 4 | 时间压缩倍率 (16→4) |
| disc_start | 50001 | 判别器启动步数 |
| kl_weight | 1e-6 | KL 散度权重 |
| disc_weight | 0.5 | 判别器损失权重 |
| num_frames | 16 | 每 clip 帧数 |
| sampling_rate | 4 | 帧间隔 |
| input_size | 224 | 空间分辨率 |
| LR 缩放 | lr × total_batch / 256 | |

## 参考

- VideoMAEv2: https://github.com/OpenGVLab/VideoMAEv2
- VideoVAEPlus: `pretrain/VideoVAEPlus/`
- VJEPA: `src/models/vision_transformer.py`
- 下游模型构建: `buildmodel.py`
