import os
import math
import warnings
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from decord import VideoReader, cpu

# 复用现有的 Transform 逻辑
from utils import make_transforms

class VideoFolderDataset(Dataset):
    """
    类似 ImageFolder 的视频数据集类。
    支持自动从按类别划分的子文件夹 (如 class_0, class_1, class_2) 中加载 mp4 视频，
    并完美保留 VJEPA 的多片段(clip)切分、抽取策略以及输出格式。
    """
    def __init__(
        self,
        root_dir,
        frames_per_clip=16,
        frame_step=2,
        num_clips=1,
        transform=None,
        shared_transform=None,
        random_clip_sampling=False, # 推理时设为 False 保证评估稳定
        allow_clip_overlap=False,
        filter_short_videos=False,
        filter_long_videos=int(10**9),
    ):
        self.root_dir = root_dir
        self.frames_per_clip = frames_per_clip
        self.frame_step = frame_step
        self.num_clips = num_clips
        self.transform = transform
        self.shared_transform = shared_transform
        self.random_clip_sampling = random_clip_sampling
        self.allow_clip_overlap = allow_clip_overlap
        self.filter_short_videos = filter_short_videos
        self.filter_long_videos = filter_long_videos

        self.samples = []
        self.labels =[]
        
        # 1. 扫描文件夹并构建类别映射 (类似 ImageFolder)
        classes = sorted([d.name for d in os.scandir(self.root_dir) 
                         if d.is_dir() and d.name.startswith('class_')])
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {self.root_dir}.")
            
        # 兼容 class_0, class_1 命名，自动映射为数字标签
        self.class_to_idx = {cls_name: int(cls_name.split('_')[-1]) if '_' in cls_name else i 
                             for i, cls_name in enumerate(classes)}

        # 2. 遍历加载所有 .mp4 文件
        for target_class in sorted(self.class_to_idx.keys()):
            class_index = self.class_to_idx[target_class]
            target_dir = os.path.join(self.root_dir, target_class)
            for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                for fname in sorted(fnames):
                    if fname.lower().endswith('.mp4'):
                        path = os.path.join(root, fname)
                        self.samples.append(path)
                        self.labels.append(class_index)

        print(f"=> Found {len(self.samples)} videos in {len(classes)} classes.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        loaded_sample = False
        
        # Keep trying to load videos until you find a valid sample (跳过损坏的视频)
        while not loaded_sample:
            loaded_sample = self.get_item_video(index)
            if not loaded_sample:
                index = np.random.randint(self.__len__())
                sample = self.samples[index]

        return loaded_sample

    def get_item_video(self, index):
        sample = self.samples[index]
        label = self.labels[index]

        # 从原来的代码完全移植
        buffer, clip_indices = self.loadvideo_decord(sample, self.frames_per_clip)
        loaded_video = len(buffer) > 0
        if not loaded_video:
            return None

        def split_into_clips(video):
            """Split video into a list of clips"""
            fpc = self.frames_per_clip
            nc = self.num_clips
            return [video[i * fpc : (i + 1) * fpc] for i in range(nc)]

        # Parse video into frames & apply data augmentations
        if self.shared_transform is not None:
            buffer = self.shared_transform(buffer)
            
        buffer = split_into_clips(buffer)
        
        if self.transform is not None:
            buffer = [self.transform(clip) for clip in buffer]

        return buffer, label, clip_indices

    def loadvideo_decord(self, sample, fpc):
        """完全保留原版的 Decord 读取逻辑"""
        fname = sample
        if not os.path.exists(fname):
            warnings.warn(f"video path not found {fname=}")
            return[], None

        _fsize = os.path.getsize(fname)
        if _fsize > self.filter_long_videos:
            warnings.warn(f"skipping long video of size {_fsize=} (bytes)")
            return[], None

        try:
            vr = VideoReader(fname, num_threads=-1, ctx=cpu(0))
        except Exception:
            return[], None

        fstp = self.frame_step
        assert fstp is not None and fstp > 0
        clip_len = int(fpc * fstp)

        if self.filter_short_videos and len(vr) < clip_len:
            warnings.warn(f"skipping short video of length {len(vr)}")
            return[], None

        vr.seek(0)
        partition_len = len(vr) // self.num_clips

        all_indices, clip_indices = [],[]
        for i in range(self.num_clips):
            if partition_len > clip_len:
                end_indx = clip_len
                if self.random_clip_sampling:
                    end_indx = np.random.randint(clip_len, partition_len)
                start_indx = end_indx - clip_len
                indices = np.linspace(start_indx, end_indx, num=fpc)
                indices = np.clip(indices, start_indx, end_indx - 1).astype(np.int64)
                indices = indices + i * partition_len
            else:
                if not self.allow_clip_overlap:
                    indices = np.linspace(0, partition_len, num=partition_len // fstp)
                    indices = np.concatenate((indices, np.ones(fpc - partition_len // fstp) * partition_len))
                    indices = np.clip(indices, 0, partition_len - 1).astype(np.int64)
                    indices = indices + i * partition_len
                else:
                    sample_len = min(clip_len, len(vr)) - 1
                    indices = np.linspace(0, sample_len, num=sample_len // fstp)
                    indices = np.concatenate((indices, np.ones(fpc - sample_len // fstp) * sample_len))
                    indices = np.clip(indices, 0, sample_len - 1).astype(np.int64)
                    clip_step = 0
                    if len(vr) > clip_len:
                        clip_step = (len(vr) - clip_len) // (self.num_clips - 1)
                    indices = indices + i * clip_step

            clip_indices.append(indices)
            all_indices.extend(list(indices))

        buffer = vr.get_batch(all_indices).asnumpy()
        return buffer, clip_indices

# ==========================================
# 1. 定义一个支持返回文件路径的 Dataset 包装器
# ==========================================
class InferenceVideoDataset(VideoFolderDataset):
    def __init__(self, root_dir, **kwargs):
        """
        root_dir: 文件夹路径列表，例如["/path/to/val", "/path/to/test"]
        """
        # 先用第一个目录初始化基类
        super().__init__(root_dir=root_dir[0], **kwargs)

        all_samples = []
        all_labels =[]

        print(f"=> 正在扫描所有目录并合并数据...")
        for d in root_dir:
            if not os.path.exists(d):
                print(f"警告: 目录不存在 {d}")
                continue
            # 临时实例化以获取该目录下的 samples 和 labels
            tmp_ds = VideoFolderDataset(root_dir=d, **kwargs)
            all_samples.extend(tmp_ds.samples)
            all_labels.extend(tmp_ds.labels)

        # 覆盖基类的属性，确保 samples 和 labels 长度一致
        self.samples = all_samples
        self.labels = all_labels
        print(f"=> 合并完成，总视频数: {len(self.samples)}")
    def __getitem__(self, index):
        sample_path = self.samples[index]
        loaded_sample = False
        
        # 保持原有逻辑：如果加载失败，随机采一个顶替
        while not loaded_sample:
            loaded_sample = self.get_item_video(index)
            if not loaded_sample:
                index = np.random.randint(self.__len__())
                sample_path = self.samples[index]  # 更新路径为实际加载成功的视频

        buffer, label, clip_indices = loaded_sample
        # 额外返回 sample_path
        return buffer, label, clip_indices, sample_path
        

# --- 1. 修复 Dataset 合并逻辑 ---
class CombinedInferenceDataset(VideoFolderDataset):
    def __init__(self, root_dirs, **kwargs):
        """
        root_dirs: 文件夹路径列表，例如["/path/to/val", "/path/to/test"]
        """
        # 先用第一个目录初始化基类
        super().__init__(root_dir=root_dirs[0], **kwargs)

        all_samples = []
        all_labels =[]

        print(f"=> 正在扫描所有目录并合并数据...")
        for d in root_dirs:
            if not os.path.exists(d):
                print(f"警告: 目录不存在 {d}")
                continue
            # 临时实例化以获取该目录下的 samples 和 labels
            tmp_ds = VideoFolderDataset(root_dir=d, **kwargs)
            all_samples.extend(tmp_ds.samples)
            all_labels.extend(tmp_ds.labels)

        # 覆盖基类的属性，确保 samples 和 labels 长度一致
        self.samples = all_samples
        self.labels = all_labels
        print(f"=> 合并完成，总视频数: {len(self.samples)}")

    def __getitem__(self, index):
        # 调用基类的视频读取函数
        # 此时 self.samples 和 self.labels 已经是完整的了
        loaded_sample = self.get_item_video(index)
        if not loaded_sample:
            return None
        buffer, label, clip_indices = loaded_sample
        sample_path = self.samples[index]
        return buffer, label, sample_path


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

def make_inference_dataloader(root_path, batch_size, world_size, rank, **kwargs):
    """
    对外暴露的 DataLoader 构建函数，与你原有的 `make_dataloader` 函数签名高度一致。
    """
    # 1. 构建 Transform (使用原工程现有的 make_transforms)
    transform = make_transforms(
        training=False, 
        num_views_per_clip=kwargs.get("num_views_per_segment", 1), 
        crop_size=kwargs.get("img_size", 224)
    )

    # 2. 实例化自定义的 ImageFolder 式 Dataset
    dataset = VideoFolderDataset(
        root_dir=root_path,
        frames_per_clip=kwargs.get("frames_per_clip", 16),
        frame_step=kwargs.get("frame_step", 2),
        num_clips=kwargs.get("num_segments", 1),
        transform=transform,
        shared_transform=None,  
        random_clip_sampling=False, # 推理时设为 False 避免随机性
    )

    # 3. 构造 DDP 的 Sampler
    sampler = DistributedSampler(
        dataset, 
        num_replicas=world_size, 
        rank=rank, 
        shuffle=False,       # 验证集不需要 Shuffle
        drop_last=False      # 推理需要跑完所有数据
    )

    # 4. 创建并返回 DataLoader
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=kwargs.get("num_workers", 8),
        pin_memory=True,
        drop_last=False,
        # PyTorch 默认的 collate_fn 已经能够正确处理我们的元组与列表拼接逻辑
    )
    
    return data_loader



# ==========================================
# 测试遍历脚本：
# ==========================================
if __name__ == "__main__":
    # === 按照你提供的 YAML 提取的默认配置 ===
    default_args = {
        "root_path": "/home/lx/alg/videos_train", 
        "batch_size": 16,                       
        "world_size": 1,                        
        "rank": 0,                              
        "img_size": 224,                        
        "frames_per_clip": 16,                  
        "frame_step": 2,                        
        "num_segments": 1,                      
        "num_views_per_segment": 1,             
        "num_workers": 8,                       
    }
    
    # --- 5. 数据加载 (使用默认参数) ---
    val_loader = make_inference_dataloader(**default_args)
    print(f"初始化 DataLoader 成功！共有 {len(val_loader)} 个 Batch。")
    
    for i, data in enumerate(val_loader):
        clips, labels, clip_indices = data
        
        print(f"\n--- Batch {i} ---")
        print(f"标签 (Labels): {labels}")
        
        # 【修正这里的打印逻辑】
        # clips 是 List[List[Tensor]]，即 [segments][views] -> Batched Tensor
        for segment_idx, segment_clips in enumerate(clips):
            # segment_clips 还是一个 list，里面是不同 view 的 Tensor
            for view_idx, view_tensor in enumerate(segment_clips):
                print(f"第 {segment_idx} 片段, 第 {view_idx} 视角数据形状: {view_tensor.shape}")
                
        # 测试读取 1 个 batch 后即退出
        break