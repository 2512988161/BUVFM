import os
import argparse
import random
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from sklearn.metrics import confusion_matrix
import decord
# decord.bridge.set_bridge('torch')
import warnings
warnings.filterwarnings("ignore")
# ==========================================
# 导入你的模块
# ==========================================
from buildmodel import build_model  
from utils import make_transforms   
from mydataset import VideoFolderDataset  

# ==========================================
# 1. 继承并包装 Dataset (支持欠采样与返回文件名)
# ==========================================
class CustomVJEPADataset(VideoFolderDataset):
    def __init__(self, root_dir, mode='train', target_size=10000, **kwargs):
        # 1. 先调用父类，完成文件夹扫描和 self.samples / self.labels 的初始化
        super().__init__(root_dir, **kwargs)
        self.mode = mode
        
        # 2. 追加欠采样逻辑
        if self.mode == 'train':
            # 将 samples 和 labels 打包在一起方便过滤
            combined = list(zip(self.samples, self.labels))
            class_0_samples =[s for s in combined if s[1] == 0]
            other_samples = [s for s in combined if s[1] != 0]
            
            if len(class_0_samples) > target_size:
                random.seed(42) # 保证多卡采样一致
                class_0_samples = random.sample(class_0_samples, target_size)
            
            self.samples_combined = class_0_samples + other_samples
            random.shuffle(self.samples_combined)
            
            # 将处理后的数据写回父类的变量中
            self.samples = [s[0] for s in self.samples_combined]
            self.labels = [s[1] for s in self.samples_combined]
            print(f"[{mode}] 欠采样完成 -> 当前数据量: {len(self.samples)} 个视频。")
        else:
            print(f"[{mode}] 验证集加载完成 -> 当前数据量: {len(self.samples)} 个视频。")

    def __getitem__(self, index):
        sample_path = self.samples[index]
        loaded_sample = False
        
        # 防崩溃重采样 (调用父类的 get_item_video)
        while not loaded_sample:
            loaded_sample = self.get_item_video(index)
            if not loaded_sample:
                index = np.random.randint(self.__len__())
                sample_path = self.samples[index]

        buffer, label, clip_indices = loaded_sample
        video_name = os.path.basename(sample_path) # 提取文件名
        
        # 相比原来多返回一个 video_name，适配 CSV 导出需求
        return buffer, label, clip_indices, video_name


# ================= 2. 参数配置 =================
def parse_args():
    parser = argparse.ArgumentParser(description="Distributed Training for VJEPA2")
    parser.add_argument('--pretrained_ckpt', type=str, required=True, help="VJEPA 预训练权重路径")
    parser.add_argument('--num_frames', type=int, default=16)
    parser.add_argument('--batch_size', type=int, default=4, help="VJEPA显存占用大，建议改小")
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=5e-5, help="微调学习率")
    parser.add_argument('--freeze_backbone', action='store_true', help="冻结 VJEPA Encoder，仅训练 Classifier")
    parser.add_argument('--com_exp', action='store_true', help="原始vjepa对比实验")
    parser.add_argument('--exp_name', type=str, default=None, help="实验名称后缀，会追加到 save_dir 和 logname")
    return parser.parse_args()


# ================= 3. 主流程 =================
def main():
    args = parse_args()
    
    train_dir = "/home/lx/alg/videos_train"
    val_dir = "/home/lx/alg/videos_val"
    
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)

    # 日志设置
    logger = logging.getLogger(__name__)
    if local_rank == 0:
        log_dir = './logs_vjepa' if not args.com_exp else "./logs_vjepa/com"
        
        os.makedirs(log_dir, exist_ok=True)
        log_name = "vjepa_frozen.log" if args.freeze_backbone else "vjepa_full.log"
        if args.exp_name is not None:
            log_name = log_name.replace('.log', f'_{args.exp_name}.log')
        # 【修改这里】：加上 force=True 强制覆盖原有配置
        logging.basicConfig(level=logging.INFO, 
                            format='%(asctime)s - %(message)s',
                            handlers=[logging.FileHandler(f"{log_dir}/{log_name}"), logging.StreamHandler()],
                            force=True)
    # --- 数据加载 ---
    transform_train = make_transforms(training=True, num_views_per_clip=1, crop_size=224)
    transform_val = make_transforms(training=False, num_views_per_clip=1, crop_size=224)

    # 实例化刚才写的包装类
    train_dataset = CustomVJEPADataset(
        root_dir=train_dir, mode='train', target_size=10000,
        frames_per_clip=args.num_frames, frame_step=2, num_clips=1, 
        transform=transform_train, random_clip_sampling=True
    )
    
    val_dataset = CustomVJEPADataset(
        root_dir=val_dir, mode='val',
        frames_per_clip=args.num_frames, frame_step=2, num_clips=1, 
        transform=transform_val, random_clip_sampling=False
    )

    train_sampler = DistributedSampler(train_dataset)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler, num_workers=8, pin_memory=True)

    # --- 模型加载 ---
    if local_rank == 0: logger.info("正在加载 VJEPA 模型...")
    encoder, classifier = build_model(
        checkpoint_path=args.pretrained_ckpt,
        resolution=224,
        frames_per_clip=args.num_frames,
        num_classes=3,
        num_heads=16,
        num_probe_blocks=1
    )

    encoder = encoder.cuda(local_rank)
    classifier = classifier.cuda(local_rank)

    if args.freeze_backbone:
        encoder.eval()
        for param in encoder.parameters():
            param.requires_grad = False
        parameters_to_update = classifier.parameters()
        if local_rank == 0: logger.info(">> 已冻结 Encoder, 仅微调 Classifier <<")
    else:
        parameters_to_update = list(encoder.parameters()) + list(classifier.parameters())
        if local_rank == 0: logger.info(">> 全量微调模式 (Full Finetuning) <<")

    # DDP 包装
    encoder = DDP(encoder, device_ids=[local_rank], find_unused_parameters=True)
    classifier = DDP(classifier, device_ids=[local_rank], find_unused_parameters=True)

    # --- 损失与优化器 ---
    counts = torch.tensor([10000, 5004, 3214], dtype=torch.float)
    weights = 1.0 / counts
    weights = weights / weights.sum() * 3
    weights = weights.cuda(local_rank)
    
    criterion = nn.CrossEntropyLoss(weight=weights).cuda(local_rank)
    optimizer = optim.AdamW(parameters_to_update, lr=args.lr, weight_decay=0.05)
    
    # 混合精度 AMP
    scaler = torch.cuda.amp.GradScaler()

    best_val_acc = 0.0
    save_dir = './ckpts/vjepa_frozen' if args.freeze_backbone else './ckpts/vjepa_full'
    if args.com_exp:
        save_dir = './ckpts/vjepa_ori'
    if args.exp_name is not None:
        save_dir = f"{save_dir}_{args.exp_name}"
    if local_rank == 0: os.makedirs(save_dir, exist_ok=True)

    # --- 训练循环 ---
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        if not args.freeze_backbone: encoder.train()
        classifier.train()
        
        train_metrics = torch.zeros(3).cuda(local_rank) 

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", disable=(local_rank != 0))
        for data in pbar:
            clips, labels, clip_indices, names = data
            labels = labels.cuda(local_rank)
            
            # 将嵌套的视频列表放到 GPU
            clips_gpu = [[dij.to(local_rank, non_blocking=True) for dij in di] for di in clips]

            optimizer.zero_grad()
            
            # AMP 混合精度前向传播
            with torch.cuda.amp.autocast(dtype=torch.float16):
                features = encoder(clips_gpu)
                logits_list = [classifier(f) for f in features]
                logits = sum(logits_list) / len(logits_list)
                loss = criterion(logits, labels)

            # AMP 反向传播
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            preds = logits.argmax(1)
            train_metrics[0] += loss.item() * labels.size(0)
            train_metrics[1] += (preds == labels).sum().item()
            train_metrics[2] += labels.size(0)

        # --- 验证循环 ---
        encoder.eval()
        classifier.eval()
        val_metrics = torch.zeros(3).cuda(local_rank)
        epoch_results = []
        all_preds =[]
        all_labels =[]

        with torch.no_grad():
            for data in val_loader:
                clips, labels, clip_indices, names = data
                labels = labels.cuda(local_rank)
                clips_gpu = [[dij.to(local_rank, non_blocking=True) for dij in di] for di in clips]

                with torch.cuda.amp.autocast(dtype=torch.float16):
                    features = encoder(clips_gpu)
                    logits_list =[classifier(f) for f in features]
                    logits = sum(logits_list) / len(logits_list)
                    loss = criterion(logits, labels)
                
                preds = logits.argmax(1)
                val_metrics[0] += loss.item() * labels.size(0)
                val_metrics[1] += (preds == labels).sum().item()
                val_metrics[2] += labels.size(0)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                probs = torch.softmax(logits, dim=1)
                for i in range(len(names)):
                    epoch_results.append({
                        'video_name': names[i],
                        'label': labels[i].item(),
                        'p0': probs[i][0].item(), 'p1': probs[i][1].item(), 'p2': probs[i][2].item()
                    })

        # --- DDP 数据同步 ---
        dist.all_reduce(train_metrics, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_metrics, op=dist.ReduceOp.SUM)
        
        global_train_loss = train_metrics[0] / train_metrics[2]
        global_train_acc  = 100. * train_metrics[1] / train_metrics[2]
        global_val_loss   = val_metrics[0] / val_metrics[2]
        global_val_acc    = 100. * val_metrics[1] / val_metrics[2]

        all_rank_results =[None for _ in range(world_size)]
        dist.all_gather_object(all_rank_results, epoch_results)
        
        all_rank_preds =[None for _ in range(world_size)]
        all_rank_labels =[None for _ in range(world_size)]
        dist.all_gather_object(all_rank_preds, all_preds)
        dist.all_gather_object(all_rank_labels, all_labels)

        # --- 日志与保存 ---
        if local_rank == 0:
            logger.info(
                f"Epoch {epoch+1:03d}/{args.epochs} | "
                f"Train Loss: {global_train_loss:.4f} | Train Acc: {global_train_acc:.2f}% | "
                f"Val Loss: {global_val_loss:.4f} | Val Acc: {global_val_acc:.2f}%"
            )
            
            flat_preds =[p for sublist in all_rank_preds for p in sublist]
            flat_labels =[l for sublist in all_rank_labels for l in sublist]
            cm = confusion_matrix(flat_labels, flat_preds)
            logger.info(f"Confusion Matrix:\n{cm}")

            if global_val_acc > best_val_acc:
                best_val_acc = global_val_acc
                
                # 【极其重要】按 VJEPA 格式打包权重，方便日后直接被 build_model 加载
                state_dict = {
                    "encoder": encoder.module.state_dict(),
                    "classifiers":[classifier.module.state_dict()],
                    "epoch": epoch
                }
                torch.save(state_dict, f"{save_dir}/best_vjepa_model.pt")
                
                # 保存评估 CSV
                combined_res =[item for sublist in all_rank_results for item in sublist]
                df = pd.DataFrame(combined_res).drop_duplicates(subset=['video_name'])
                df.to_csv(f"{save_dir}/best_vjepa_eval.csv", index=False)
                logger.info(f"--> Saved Best Model ({best_val_acc:.2f}%)")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()