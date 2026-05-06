import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 必须在导入 plt 前执行，防止在无显示器的服务器上卡死
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# 导入你现有的模块
from buildmodel import build_model
from mydataset import CombinedInferenceDataset
from utils import make_transforms



def parse_patient_id(filename):
    """
    解析文件名：0d90a1767fc4457594006399175fa582#Patient#66.mp4
    返回：Patient #66
    """
    try:
        # 去掉后缀并按 # 分割
        parts = filename.replace(".mp4", "").split('#')
        if len(parts) >= 3:
            # 拼接 Patient 和 ID，例如 Patient #66
            return f"{parts[1]} #{parts[2]}"
        return "Unknown"
    except:
        return "Unknown"

def main():
    # --- 配置参数 ---
    # checkpoint_path = "/home/lx/baselines/vjepa/ckpts/vjepa_full_vaee1/best_vjepa_model.pt"
    checkpoint_path = "/home/lx/baselines/vjepa/ckpts/vjepa_full/best_vjepa_model9639(paper).pt"
    data_dirs = ["/home/lx/dataset/huigu_internal_0506"]
    tsne_dir = os.path.join('output', "tsne_internal0506",'ours')
    # data_dirs =["/home/lx/alg/videos_val", "/home/lx/alg/videos_test"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args = {
        "batch_size": 4,      # batch 改小点防止显存溢出
        "img_size": 224,
        "frames_per_clip": 16,
        "frame_step": 2,
        "num_segments": 1,
        "num_workers": 0,     # 如果之前卡死，强烈建议先设为 0 调试，稳定后再改大
    }

    # --- 2. 加载模型 ---
    print("=> 正在加载模型...")
    encoder, _ = build_model(
        checkpoint_path=checkpoint_path,
        resolution=args["img_size"],
        frames_per_clip=args["frames_per_clip"],
        num_classes=3
    )
    encoder = encoder.to(device).eval()

    # --- 3. 准备数据加载器 ---
    transform = make_transforms(training=False, crop_size=args["img_size"])
    dataset = CombinedInferenceDataset(
        root_dirs=data_dirs,
        frames_per_clip=args["frames_per_clip"],
        frame_step=args["frame_step"],
        num_clips=args["num_segments"],
        transform=transform
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args["batch_size"],
        shuffle=False, num_workers=args["num_workers"],
        pin_memory=True
    )

    # --- 4. 提取特征 ---
    all_embeddings =[]
    all_video_names = []
    all_patient_ids = []
    all_labels =[]

    print("=> 开始提取特征...")
    with torch.no_grad():
        for batch in tqdm(dataloader):
            if batch is None: continue
            clips, labels, paths = batch

            # 适配 V-JEPA 的输入格式 list(Tensor[B, C, T, H, W])
            clips_gpu = [[view.to(device) for view in segment] for segment in clips]

            with torch.cuda.amp.autocast(dtype=torch.float16):
                features_list = encoder(clips_gpu)
                feat = features_list[0] # [B, N, D]
                pooled_feat = feat.mean(dim=1) # 空间+时间池化 ->[B, D]

            all_embeddings.append(pooled_feat.cpu().numpy())

            # 保存真实标签，处理可能为 Tensor 的情况
            if torch.is_tensor(labels):
                all_labels.extend(labels.cpu().tolist())
            else:
                all_labels.extend(labels)

            for p in paths:
                fname = os.path.basename(p)
                all_video_names.append(fname)
                all_patient_ids.append(parse_patient_id(fname))

    # 合并成大矩阵并释放显存
    embeddings = np.concatenate(all_embeddings, axis=0)
    del encoder
    torch.cuda.empty_cache()

    # --- 5. 准备输出目录 ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    os.makedirs(tsne_dir, exist_ok=True)

    print(f"\n=> 总样本数: {len(all_video_names)}")

    # --- 6. 保存大的 features.npz ---
    np.savez(os.path.join(tsne_dir, "features.npz"),
             embeddings=embeddings,
             video_names=np.array(all_video_names),
             patient_ids=np.array(all_patient_ids),
             labels=np.array(all_labels))
    print(f"  特征已保存至 {tsne_dir}/features.npz")

    # --- 7. 预处理与降维 ---
    scaler = StandardScaler()
    emb_scaled = scaler.fit_transform(embeddings)

    pca_dim = min(50, len(emb_scaled))
    pca = PCA(n_components=pca_dim)
    feat_pca = pca.fit_transform(emb_scaled)

    # --- 8. t-SNE 可视化 ---
    print(f"  正在生成 t-SNE 图...")
    perp = min(30, len(feat_pca) - 1)
    tsne = TSNE(n_components=2, random_state=42, perplexity=perp)
    feat_2d = tsne.fit_transform(feat_pca)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(feat_2d[:, 0], feat_2d[:, 1], c=all_labels, cmap='tab10', alpha=0.7)
    handles, classes = scatter.legend_elements()
    plt.legend(handles, classes, title="Classes")
    plt.title("VJEPA Feature t-SNE (Ground Truth)")
    plt.savefig(os.path.join(tsne_dir, "tsne.png"))
    plt.close()

    print(f"\n=> 任务完成！结果已保存至 {tsne_dir}/")

if __name__ == "__main__":
    main()
