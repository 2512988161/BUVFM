import os
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

# 导入你现有的模块
from buildmodel import build_model  
from utils import make_transforms 
from decord import VideoReader, cpu
import warnings

# ==========================================
# 1. 定义一个通用的推理 Dataset
# ==========================================
class SimpleVideoDataset(Dataset):
    """
    专门用于推理的 Dataset：递归扫描目录下所有 mp4，不要求 class_ 命名
    """
    def __init__(self, root_dir, frames_per_clip=16, frame_step=2, num_clips=1, transform=None):
        self.root_dir = root_dir
        self.frames_per_clip = frames_per_clip
        self.frame_step = frame_step
        self.num_clips = num_clips
        self.transform = transform
        
        self.samples = []
        for root, _, fnames in os.walk(root_dir):
            for fname in fnames:
                if fname.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    self.samples.append(os.path.join(root, fname))
        
        if not self.samples:
            print(f"警告: 在 {root_dir} 中没找到任何视频文件！")

    def __len__(self):
        return len(self.samples)

    def loadvideo_decord(self, sample):
        try:
            vr = VideoReader(sample, num_threads=-1, ctx=cpu(0))
            if len(vr) < self.frames_per_clip * self.frame_step:
                # 视频太短，简单处理：直接采样前几帧
                indices = np.linspace(0, len(vr) - 1, num=self.frames_per_clip).astype(np.int64)
            else:
                # 采样中间段落
                clip_len = self.frames_per_clip * self.frame_step
                start_idx = (len(vr) - clip_len) // 2
                indices = np.arange(start_idx, start_idx + clip_len, self.frame_step)[:self.frames_per_clip]
            
            buffer = vr.get_batch(indices).asnumpy()
            return buffer
        except Exception as e:
            print(f"读取失败 {sample}: {e}")
            return None

    def __getitem__(self, index):
        path = self.samples[index]
        buffer = self.loadvideo_decord(path)
        
        if buffer is None:
            # 如果读取失败，返回一个全零 Tensor 占位，后面逻辑会处理
            return torch.zeros(self.num_clips, 3, self.frames_per_clip, 224, 224), path, False

        # Transform 处理
        # 注意：此处假设 num_clips=1, 仿照你原有的 split_into_clips 逻辑
        if self.transform:
            # 这里的 transform 通常期望 [T, H, W, C]
            # 为了适配 make_transforms，我们将 buffer 包装成 list
            processed_clips = [self.transform(buffer)] 
            
        return processed_clips, path, True

# ==========================================
# 2. 主推理函数
# ==========================================
def run_inference():
    # --- 配置参数 ---
    checkpoint_path = "/home/lx/baselines/vjepa/ckpts/vjepa_full/best_vjepa_model9639(paper).pt"
    input_dir = "/home/wcz/workspace/DATASET/pros_dataset_cloud_patient_case"
    output_csv = "./csvs/test_results.csv"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    args = {
        "batch_size": 8, # 根据显存调整
        "img_size": 224,
        "frames_per_clip": 16,
        "frame_step": 2,
        "num_segments": 1,
        "num_views_per_segment": 1,
        "num_workers": 4,
    }

    # --- 1. 加载模型 ---
    print(f"=> 加载权重: {checkpoint_path}")
    model, classifier = build_model(
        checkpoint_path=checkpoint_path,
        resolution=args["img_size"],
        frames_per_clip=args["frames_per_clip"],
        num_classes=3,
        num_heads=16,
        num_probe_blocks=1
    )
    model = model.to(device).eval()
    classifier = classifier.to(device).eval()

    # --- 2. 准备数据 ---
    transform = make_transforms(
        training=False, 
        num_views_per_clip=args["num_views_per_segment"], 
        crop_size=args["img_size"]
    )
    
    dataset = SimpleVideoDataset(
        root_dir=input_dir,
        frames_per_clip=args["frames_per_clip"],
        frame_step=args["frame_step"],
        num_clips=args["num_segments"],
        transform=transform
    )
    
    loader = DataLoader(dataset, batch_size=args["batch_size"], shuffle=False, num_workers=args["num_workers"])

    # --- 3. 执行推理 ---
    results = []
    print(f"=> 开始推理，视频总数: {len(dataset)}")

    with torch.no_grad():
        for clips, paths, valid_mask in tqdm(loader):
            # 过滤无效视频
            if not any(valid_mask): continue
            
            # 只有有效的视频才送入模型
            # clips 结构是 List[List[Tensor]] -> [view][batch, C, T, H, W]
            # 数据转换 (处理嵌套 List)
            clips_gpu = [[v.to(device, non_blocking=True) for v in view_list] for view_list in clips]

            with torch.amp.autocast('cuda', dtype=torch.float16):
                # 提取特征
                features = model(clips_gpu)
                # 分类
                logits_list = [classifier(f) for f in features]
                # 集成 views (求平均)
                probs = sum([F.softmax(l, dim=1) for l in logits_list]) / len(logits_list)
            
            probs_cpu = probs.cpu().numpy()
            preds = probs_cpu.argmax(axis=1)

            for i in range(len(paths)):
                if valid_mask[i]:
                    results.append({
                        "video_name": os.path.basename(paths[i]),
                        "p0": f"{probs_cpu[i][0]:.4f}",
                        "p1": f"{probs_cpu[i][1]:.4f}",
                        "p2": f"{probs_cpu[i][2]:.4f}",
                        "pre": preds[i]
                    })

    # --- 4. 保存结果 ---
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"\n=> 推理完成！结果已保存至: {output_csv}")
    print(df.head())

if __name__ == "__main__":
    run_inference()