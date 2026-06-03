import os
import sys
import cv2
import json
import time
import random
import argparse
import subprocess
import threading
import queue
import gc
import re
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from PIL import Image
from torchvision import transforms
import timm  # 新增依赖
import shutil

_VJEPA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _VJEPA_ROOT not in sys.path:
    sys.path.insert(0, _VJEPA_ROOT)

from distill.cls.buildmodel import MobileNetV3Backbone


# ---------------------------------------------------------------------------
# Reusable model builders (also imported by pipeline_utils.py)
# ---------------------------------------------------------------------------

def build_mobilenet_svm(checkpoint_path, model_name="mobilenetv3_small_075",
                        num_classes=2, device="cpu"):
    """Build MobileNetV3Backbone, load SVM checkpoint, return eval model.

    Falls back to plain timm model if MobileNetV3Backbone construction fails.
    The returned model always has a ``forward_full(x)`` method that returns logits.
    """
    try:
        model = MobileNetV3Backbone(model_name=model_name, num_classes=num_classes).to(device)
        if os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location=device)
            state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
            model.load_state_dict(state_dict)
        model.eval()
        return model
    except Exception:
        model = timm.create_model(model_name, pretrained=False, in_chans=3,
                                  num_classes=num_classes).to(device)
        if os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location=device)
            state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
            model.load_state_dict(state_dict)
        model.forward_full = model.forward
        model.eval()
        return model


def get_svm_preprocess():
    """Return standard SVM/MobileNet preprocessing transform (ImageNet stats)."""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])


# ================= 1. 环境设置 =================
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
cv2.setNumThreads(0)

# ================= 3. 可视化函数 =================
def visualize_video(all_frames, video_frame_results, yolo_model, width, height, fps, out_path, device):
    """在完整视频上可视化 YOLO class-0 检测框和 MobileNet 分类分数。"""
    from ultralytics import YOLO

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    total_frames = len(all_frames)
    for i in range(0, total_frames, BATCH_SIZE):
        batch_frames = all_frames[i : i + BATCH_SIZE]
        curr_batch_len = len(batch_frames)

        # YOLO 推理所有帧
        y_results = yolo_model(batch_frames, imgsz=YOLO_IMG_SIZE, conf=YOLO_CONF_THRESHOLD,
                               device=device, verbose=False)

        for b_idx in range(curr_batch_len):
            abs_idx = i + b_idx
            frame = batch_frames[b_idx].copy()
            mb_score = video_frame_results[abs_idx]["mobilenet_score"]

            # YOLO class-0 红框
            res = y_results[b_idx]
            if res.boxes is not None:
                for box in res.boxes:
                    cls_id = int(box.cls[0])
                    if cls_id != 0:  # 只可视化 0 类
                        continue
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, f"{conf:.2f}", (x1, max(y1 - 5, 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # MobileNet 分数在右上角
            text = f"MB: {mb_score:.3f}"
            color = (0, 255, 0) if mb_score < 0.5 else (0, 0, 255)
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.putText(frame, text, (width - tw - 10, th + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            writer.write(frame)

    writer.release()

# ================= 4. 视频读取类 (后台线程) =================
class BackgroundFrameReader(threading.Thread):
    def __init__(self, video_path, batch_size, queue_size=4):
        super().__init__()
        self.video_path = video_path
        self.batch_size = batch_size
        self.queue = queue.Queue(maxsize=queue_size)
        self.stopped = False
        self.video_info = {}

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.queue.put(None)
            return
        
        self.video_info = {
            'width': int(cap.get(3)), 
            'height': int(cap.get(4)), 
            'fps': cap.get(5),
            'total_frames': int(cap.get(7))
        }
        
        frames_buffer = []
        while not self.stopped:
            ret, frame = cap.read()
            if ret:
                frames_buffer.append(frame)
                if len(frames_buffer) == self.batch_size:
                    self.queue.put(frames_buffer)
                    frames_buffer = []
            else:
                if frames_buffer: self.queue.put(frames_buffer)
                break
        cap.release()
        self.queue.put(None)

    def stop(self):
        self.stopped = True

# ================= 5. Worker 逻辑 (核心处理：双模型串行与Clip提取) =================
def run_worker(chunk_json_path, worker_id):
    # 延迟导入 YOLO 以防止多进程冲突
    from ultralytics import YOLO

    time.sleep(worker_id * 2.0) # 错峰启动
    
    with open(chunk_json_path, 'r') as f: video_paths = json.load(f)
    
    # 显卡设置：由 Master 通过 CUDA_VISIBLE_DEVICES 控制，这里统一用 cuda:0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[Worker {worker_id}] Initializing models on {device}...", flush=True)

    # --- A. 加载 YOLO 模型 ---
    try:
        yolo_model = YOLO(YOLO_MODEL_PATH)
        yolo_model(np.zeros((YOLO_IMG_SIZE, YOLO_IMG_SIZE, 3), dtype=np.uint8), device=device, verbose=False)
        print(f"[Worker {worker_id}] YOLO loaded.", flush=True)
    except Exception as e:
        print(f"[Worker {worker_id}] Failed to load YOLO: {e}", flush=True)
        return

    # --- B. 加载 MobileNet (SVM) 模型 ---
    try:
        svm_model = build_mobilenet_svm(SVM_MODEL_PATH, device=device)
        print(f"[Worker {worker_id}] MobileNetV3 loaded.", flush=True)
    except Exception as e:
        print(f"[Worker {worker_id}] Failed to load MobileNet: {e}", flush=True)
        return

    # --- C. MobileNet 预处理定义 ---
    svm_preprocess = get_svm_preprocess()
    worker_results = {}     
    worker_lesion_clip_stats = {} # 专门记录每个视频包含的病灶 Clip 数量
    
    for v_idx, video_path in enumerate(video_paths):
        t_start = time.time()
        filename = os.path.basename(video_path)
        base_filename = os.path.splitext(filename)[0]
        rel_path = os.path.relpath(video_path, INPUT_FOLDER)
        rel_dir = os.path.dirname(rel_path)
        clip_subfolder = os.path.join(CLIP_SAVE_DIR, rel_dir, base_filename)
        os.makedirs(clip_subfolder, exist_ok=True)
        
        video_frame_results = []
        
        # 为了进行全局排序和填充，全视频读入
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): continue
        width = int(cap.get(3))
        height = int(cap.get(4))
        fps = cap.get(5)
        all_frames = []
        while True:
            ret, frame = cap.read()
            if not ret: break
            all_frames.append(frame)
        cap.release()
        
        total_frames = len(all_frames)
        if total_frames == 0: continue

        # --- 步骤 1: 全视频扫描，记录得分并寻找候选 ---
        consecutive_hits = 0
        current_conf_sum = 0.0
        lesion_candidates = []

        for i in range(0, total_frames, BATCH_SIZE):
            batch_frames = all_frames[i : i + BATCH_SIZE]
            curr_batch_len = len(batch_frames)
            
            # MobileNet 推理
            imgs_pil = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in batch_frames]
            svm_tensors = torch.stack([svm_preprocess(img) for img in imgs_pil]).to(device)
            with torch.inference_mode():
                logits = svm_model.forward_full(svm_tensors)
                probs = F.softmax(logits, dim=1)
                mb_scores = probs[:, 1].cpu().tolist()

            # YOLO 串行推理
            yolo_scores = [0.0] * curr_batch_len
            y_indices = [idx for idx, score in enumerate(mb_scores) if score < 0.5]
            if y_indices:
                y_inputs = [batch_frames[idx] for idx in y_indices]
                y_results = yolo_model(y_inputs, imgsz=YOLO_IMG_SIZE, conf=YOLO_CONF_THRESHOLD, device=device, verbose=False)
                for res_idx, batch_idx in enumerate(y_indices):
                    res = y_results[res_idx]
                    if res.boxes is not None:
                        confs = [float(b.conf[0]) for b in res.boxes if int(b.cls[0]) != YOLO_EXCLUDE_CLASS]
                        yolo_scores[batch_idx] = max(confs) if confs else 0.0

            for b_idx in range(curr_batch_len):
                abs_idx = i + b_idx
                y_score = yolo_scores[b_idx]
                video_frame_results.append({
                    "frame_idx": abs_idx,
                    "yolo_score": round(y_score, 4),
                    "mobilenet_score": round(mb_scores[b_idx], 4)
                })

                if y_score >= YOLO_CONF_THRESHOLD:
                    consecutive_hits += 1
                    current_conf_sum += y_score
                    if consecutive_hits == CONSECUTIVE_FRAMES:
                        start_f = max(0, abs_idx - CONSECUTIVE_FRAMES + 1)
                        lesion_candidates.append({
                            "start": start_f,
                            "avg_conf": current_conf_sum / CONSECUTIVE_FRAMES
                        })
                else:
                    consecutive_hits = 0
                    current_conf_sum = 0.0

        # --- 步骤 2: 筛选与填充逻辑 (核心修改：防止重叠) ---
        target_clips = []
        lesion_candidates.sort(key=lambda x: x["avg_conf"], reverse=True)
        lesion_starts = [] # 记录已选 Lesion 的起始位置
        lesion_clip_count = 0

        # 1. 首先选出完全不重叠的 Lesion 片段
        for cand in lesion_candidates:
            if len(target_clips) >= 6: break
            # 保证 Lesion 片段之间完全不重叠 (阈值设为 60)
            if any(abs(cand["start"] - s) < 60 for s in lesion_starts): continue
            
            target_clips.append({"start": cand["start"], "type": "Lesion"})
            lesion_starts.append(cand["start"])
            lesion_clip_count += 1

        # 2. 如果不够 6 个，填充 NonLesion 片段，且不能与 Lesion 片段重叠
        if len(target_clips) < 6:
            needed = 6 - len(target_clips)
            if total_frames > 60:
                step = total_frames // (needed + 1)
                for k in range(1, needed + 1):
                    fill_start = k * step
                    # 检查是否与任何已选的 Lesion 片段重叠
                    # 如果该 NonLesion 起始点落在任何 Lesion 的 [start-60, start+60] 范围内，则判定为重叠
                    is_overlap_with_lesion = any(abs(fill_start - s) < 60 for s in lesion_starts)
                    
                    if is_overlap_with_lesion:
                        # 如果冲突，尝试寻找一个不冲突的位置（这里简单地向后平移直到不冲突或出界）
                        found_safe = False
                        for offset in range(60, total_frames, 60):
                            new_fill = (fill_start + offset) % (total_frames - 60)
                            if not any(abs(new_fill - s) < 60 for s in lesion_starts):
                                fill_start = new_fill
                                found_safe = True
                                break
                        if not found_safe: fill_start = 0 # 兜底

                    target_clips.append({"start": fill_start, "type": "NonLesion"})
            else:
                for _ in range(needed): target_clips.append({"start": 0, "type": "NonLesion"})

        # --- 步骤 3: 物理保存 ---
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        for c_idx, clip_info in enumerate(target_clips[:6]):
            prefix = clip_info["type"] + "_"
            start_f = clip_info["start"]
            clip_name = f"{prefix}clip_{c_idx+1:02d}_start_{start_f}.mp4"
            clip_path = os.path.join(clip_subfolder, clip_name)
            out_writer = cv2.VideoWriter(clip_path, fourcc, fps, (width, height))
            for f_offset in range(60):
                read_idx = start_f + f_offset
                out_writer.write(all_frames[read_idx] if read_idx < total_frames else all_frames[-1])
            out_writer.release()

        worker_results[filename] = video_frame_results
        worker_lesion_clip_stats[filename] = lesion_clip_count # 存入该视频病灶Clip数量
        
        fps_rate = total_frames / (time.time() - t_start + 1e-6)
        print(f"[Worker {worker_id}] {rel_path} processed, Lesion Clips: {lesion_clip_count}", flush=True)

        # --- 可视化输出 (可选) ---
        if VIDEO_OUTPUT_DIR:
            viz_dir = os.path.join(VIDEO_OUTPUT_DIR, rel_dir)
            os.makedirs(viz_dir, exist_ok=True)
            viz_path = os.path.join(viz_dir, base_filename + ".mp4")
            visualize_video(all_frames, video_frame_results, yolo_model, width, height, fps, viz_path, device)

        gc.collect()
        torch.cuda.empty_cache()

    # 保存结果
    with open(f"temp_result_worker_{worker_id}.json", 'w') as f: json.dump(worker_results, f)
    with open(f"temp_stats_worker_{worker_id}.json", 'w') as f: json.dump(worker_lesion_clip_stats, f)

# ================= 6. Master 逻辑 (管理与分发) =================
def run_master(args):
    target_class = args.target_class
    print("="*60)
    print(f"【Cascaded Model Inference: MobileNetV3 -> YOLO】")
    print(f"Target Class: {target_class}")
    print("="*60)
    
    if os.path.exists(CLIP_SAVE_DIR): shutil.rmtree(CLIP_SAVE_DIR) 
    os.makedirs(CLIP_SAVE_DIR, exist_ok=True)

    video_paths = []
    exts = {'.mp4', '.wmv', '.avi', '.mov', '.mkv', '.MP4', '.WMV', '.AVI'}
    scan_dir = INPUT_FOLDER if target_class == "all" else os.path.join(INPUT_FOLDER, target_class)
    for root, _, files in os.walk(scan_dir):
        for f in files:
            if any(f.endswith(e) for e in exts): video_paths.append(os.path.join(root, f))
    
    if not video_paths: return
    random.shuffle(video_paths)
    total_workers = NUM_GPUS * WORKERS_PER_GPU
    chunks = np.array_split(video_paths, total_workers)
    processes = []
    
    try:
        for i in range(total_workers):
            if len(chunks[i]) == 0: continue
            chunk_file = f"temp_chunk_dual_{i}.json"
            with open(chunk_file, 'w') as f: json.dump(chunks[i].tolist(), f)
            env = os.environ.copy()
            env["PYTHONPATH"] = _VJEPA_ROOT + os.pathsep + env.get("PYTHONPATH", "")
            env["CUDA_VISIBLE_DEVICES"] = str(i // WORKERS_PER_GPU)
            cmd = [
                sys.executable, os.path.abspath(__file__),
                "--worker_mode", "--worker_id", str(i), "--chunk_json", chunk_file,
                "--input_folder", INPUT_FOLDER,
                "--clip_save_dir", CLIP_SAVE_DIR,
                "--yolo_model_path", YOLO_MODEL_PATH,
                "--svm_model_path", SVM_MODEL_PATH,
                "--batch_size", str(BATCH_SIZE),
                "--yolo_img_size", str(YOLO_IMG_SIZE),
                "--yolo_exclude_class", str(YOLO_EXCLUDE_CLASS),
                "--yolo_conf_threshold", str(YOLO_CONF_THRESHOLD),
                "--consecutive_frames", str(CONSECUTIVE_FRAMES),
                "--video_output_dir", VIDEO_OUTPUT_DIR,
            ]
            processes.append(subprocess.Popen(cmd, env=env))
        for p in processes: p.wait()
    except KeyboardInterrupt:
        for p in processes: p.terminate()

    # 汇总
    final_results = {}
    all_lesion_stats = {}
    for i in range(total_workers):
        res_file = f"temp_result_worker_{i}.json"
        stats_file = f"temp_stats_worker_{i}.json"
        if os.path.exists(res_file):
            with open(res_file, 'r') as f: final_results.update(json.load(f))
            os.remove(res_file)
        if os.path.exists(stats_file):
            with open(stats_file, 'r') as f: all_lesion_stats.update(json.load(f))
            os.remove(stats_file)
        chunk_f = f"temp_chunk_dual_{i}.json"
        if os.path.exists(chunk_f): os.remove(chunk_f)

    with open(OUTPUT_JSON_NAME, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=4)
    
    # --- 打印包含病灶的视频比例报告 ---
    total_videos = len(all_lesion_stats)
    # 统计有多少个视频的病灶 Clip 数量大于 0
    videos_with_lesion = sum(1 for v in all_lesion_stats.values() if v > 0)
    lesion_ratio = (videos_with_lesion / total_videos * 100) if total_videos > 0 else 0
    
    print("\n" + "="*60)
    print("【Final Extraction Report】")
    print(f"Total Videos Processed: {total_videos}")
    print(f"Videos containing Lesion Clips: {videos_with_lesion}")
    print(f"Percentage of Videos with Lesion: {lesion_ratio:.2f}%")
    print("="*60 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker_mode", action="store_true")
    parser.add_argument("--worker_id", type=int)
    parser.add_argument("--chunk_json", type=str)
    parser.add_argument("--target_class", type=str, default="all")

    # --- 输入/输出路径 ---
    parser.add_argument("--input_folder", type=str,
                        default="/home/lx/dataset/reader_0517_orilong")
    parser.add_argument("--output_json_name", type=str,
                        default="output/QC/test.json",
                        help="最终汇总的JSON文件名")
    parser.add_argument("--clip_save_dir", type=str, default="output/QC/clip_save",
                        help="提取的clip保存主文件夹")

    # --- 模型路径 ---
    parser.add_argument("--yolo_model_path", type=str,
                        default='distill/det/output/ckpts/distilled.pt')
    parser.add_argument("--svm_model_path", type=str,
                        default='distill/cls/output/ckpts/best_finetune_distilled.pt',
                        help="MobileNet (SVM) 路径")

    # --- 硬件参数 ---
    parser.add_argument("--num_gpus", type=int, default=4)
    parser.add_argument("--workers_per_gpu", type=int, default=1,
                        help="两个模型同时运行显存压力大，建议每张卡1个Worker")
    parser.add_argument("--batch_size", type=int, default=16)

    # --- YOLO & Clip 提取参数 ---
    parser.add_argument("--yolo_img_size", type=int, default=640)
    parser.add_argument("--yolo_exclude_class", type=int, default=1,
                        help="YOLO 排除的类别ID")
    parser.add_argument("--yolo_conf_threshold", type=float, default=0.5,
                        help="YOLO 判定有病灶的置信度阈值")
    parser.add_argument("--consecutive_frames", type=int, default=8,
                        help="触发保存机制的连续检测帧数")
    parser.add_argument("--video_output_dir", type=str, default="",
                        help="可视化视频输出目录，留空则不输出")

    args = parser.parse_args()

    # 将配置赋值为模块全局变量, 供 run_worker / run_master 内部引用
    INPUT_FOLDER = args.input_folder
    OUTPUT_JSON_NAME = args.output_json_name
    CLIP_SAVE_DIR = args.clip_save_dir
    YOLO_MODEL_PATH = args.yolo_model_path
    SVM_MODEL_PATH = args.svm_model_path
    NUM_GPUS = args.num_gpus
    WORKERS_PER_GPU = args.workers_per_gpu
    BATCH_SIZE = args.batch_size
    YOLO_IMG_SIZE = args.yolo_img_size
    YOLO_EXCLUDE_CLASS = args.yolo_exclude_class
    YOLO_CONF_THRESHOLD = args.yolo_conf_threshold
    CONSECUTIVE_FRAMES = args.consecutive_frames
    VIDEO_OUTPUT_DIR = args.video_output_dir

    if args.worker_mode:
        run_worker(args.chunk_json, args.worker_id)
    else:
        run_master(args)