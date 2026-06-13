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


# 蒸馏版 YOLO 权重是用自定义 distill 工程保存的，
# Ultralytics 在 torch.load(yolo_distill_260612.pt) 时必须能 import distill。
# 因此这里把当前脚本目录、当前工作目录、它们的上级目录都加入 sys.path，
# 避免 YOLO 权重加载时误触发 pip install distill。
_THIS_DIR = os.path.abspath(os.path.dirname(__file__))
_CWD = os.path.abspath(os.getcwd())
for _p in [
    _THIS_DIR,
    _CWD,
    os.path.abspath(os.path.join(_THIS_DIR, '..')),
    os.path.abspath(os.path.join(_THIS_DIR, '../..')),
    os.path.abspath(os.path.join(_CWD, '..')),
    os.path.abspath(os.path.join(_CWD, '../..')),
    '/root/workspace/196_code',
    '/root/workspace/196_code/wh_data_auto_label',
]:
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

try:
    import distill  # noqa: F401  # 提前验证，YOLO 反序列化蒸馏权重时需要
except ImportError:
    print(
        "[Warning] Cannot import custom package 'distill'. "
        "If YOLO loading still fails, set PYTHONPATH to the directory that contains the distill/ folder.",
        flush=True,
    )

# ================= 1. 环境设置 =================
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
cv2.setNumThreads(0)

# ================= 2. 参数解析 =================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Video quality control: YOLO + MobileNet clip extraction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # --- Master/Worker ---
    parser.add_argument("--worker_mode", action="store_true")
    parser.add_argument("--worker_id", type=int)
    parser.add_argument("--chunk_json", type=str)
    parser.add_argument("--target_class", type=str, default="all")
    
    # --- 输入/输出路径 ---
    io = parser.add_argument_group("I/O paths")
    io.add_argument(
        "--input_folder",
        type=str,
        default="/home/wcz/workspace/DATASET/reader_study_videos_desensitization_class_0517",
        help="Input video folder",
    )
    io.add_argument("--output_json_name",type=str,default="output/QC/out.json",help="Output JSON filename",)

    # --- 模型路径 ---
    model = parser.add_argument_group("Model paths")
    model.add_argument("--yolo_model_path",type=str,default="./distill/det/output/ckpts/distilled.pt",help="YOLO distilled model path",    )
    model.add_argument( "--svm_model_path", type=str, default="./distill/cls/output/ckpts/mobilenetv3_small_075_yl_241222.pth", help="SVM (MobileNetV3) model path",    )

    # --- 硬件参数 ---
    hw = parser.add_argument_group("Hardware")
    hw.add_argument("--num_gpus",type=int,default=4,help="Number of GPUs",)
    hw.add_argument("--workers_per_gpu",type=int,default=1,help="Number of workers per GPU",  )
    hw.add_argument("--batch_size",type=int,default=16,help="Batch size for inference",  )

    # --- YOLO & Clip 提取参数 ---
    yolo = parser.add_argument_group("YOLO & Clip")
    yolo.add_argument("--yolo_img_size",type=int,default=256,help="YOLO input image size",)
    yolo.add_argument("--yolo_exclude_class",type=int,default=1,help="YOLO class ID to exclude",)
    yolo.add_argument("--yolo_conf_threshold",type=float,default=0.5,help="YOLO confidence threshold",)
    yolo.add_argument("--consecutive_frames",type=int,default=8,help="Number of consecutive frames for lesion detection",)
    yolo.add_argument("--clip_save_dir",type=str,default=None,help="Clip save directory (default: auto from yolo-conf-threshold)",)



    args = parser.parse_args()

    # --- Auto path derivation ---
    if args.clip_save_dir is None:
        args.clip_save_dir = "reader_study_clip_save_0612_distill_old_new_%f" % args.yolo_conf_threshold

    return args

def get_svm_preprocess():
    """Return standard SVM/MobileNet preprocessing transform (ImageNet stats)."""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
# ================= 3. 视频读取类 =================
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

# ================= 4. Worker 逻辑 =================
def run_worker(chunk_json_path, worker_id, args):
    from ultralytics import YOLO
    time.sleep(worker_id * 2.0)

    with open(chunk_json_path, 'r') as f: video_paths = json.load(f)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 加载模型
    yolo_model = YOLO(args.yolo_model_path)
    yolo_model(np.zeros((args.yolo_img_size, args.yolo_img_size, 3), dtype=np.uint8), device=device, verbose=False)

    pretrain_model = "mobilenetv3_small_075"
    svm_model = timm.create_model(pretrain_model, pretrained=False, in_chans=3, num_classes=2).to(device)
    if os.path.exists(args.svm_model_path):
        svm_model.load_state_dict(torch.load(args.svm_model_path, map_location=device))
    svm_model.eval()

    svm_preprocess = get_svm_preprocess()

    worker_results = {}
    worker_lesion_clip_stats = {} # 统计：video_path -> lesion_clip_count

    for v_idx, video_path in enumerate(video_paths):
        t_start = time.time()
        filename = os.path.basename(video_path)
        base_filename = os.path.splitext(filename)[0]
        rel_path = os.path.relpath(video_path, args.input_folder)
        rel_dir = os.path.dirname(rel_path)
        clip_subfolder = os.path.join(args.clip_save_dir, rel_dir, base_filename)
        os.makedirs(clip_subfolder, exist_ok=True)

        video_frame_results = []
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): continue
        width, height, fps = int(cap.get(3)), int(cap.get(4)), cap.get(5)
        all_frames = []
        while True:
            ret, frame = cap.read()
            if not ret: break
            all_frames.append(frame)
        cap.release()
        total_frames = len(all_frames)
        if total_frames == 0: continue

        # 推理逻辑
        consecutive_hits = 0
        current_conf_sum = 0.0
        lesion_candidates = []

        for i in range(0, total_frames, args.batch_size):
            batch_frames = all_frames[i : i + args.batch_size]
            curr_batch_len = len(batch_frames)
            imgs_pil = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in batch_frames]
            svm_tensors = torch.stack([svm_preprocess(img) for img in imgs_pil]).to(device)
            with torch.inference_mode():
                logits = svm_model(svm_tensors)
                probs = F.softmax(logits, dim=1)
                mb_scores = probs[:, 1].cpu().tolist()

            yolo_scores = [0.0] * curr_batch_len
            y_indices = [idx for idx, score in enumerate(mb_scores) if score < 0.5]
            if y_indices:
                y_inputs = [batch_frames[idx] for idx in y_indices]
                y_results = yolo_model(y_inputs, imgsz=args.yolo_img_size, conf=args.yolo_conf_threshold, device=device, verbose=False)
                for res_idx, batch_idx in enumerate(y_indices):
                    res = y_results[res_idx]
                    if res.boxes is not None:
                        confs = [float(b.conf[0]) for b in res.boxes if int(b.cls[0]) != args.yolo_exclude_class]
                        yolo_scores[batch_idx] = max(confs) if confs else 0.0

            for b_idx in range(curr_batch_len):
                abs_idx = i + b_idx
                y_score = yolo_scores[b_idx]
                video_frame_results.append({
                    "frame_idx": abs_idx, "yolo_score": round(y_score, 4), "mobilenet_score": round(mb_scores[b_idx], 4)
                })
                if y_score >= args.yolo_conf_threshold:
                    consecutive_hits += 1
                    current_conf_sum += y_score
                    if consecutive_hits == args.consecutive_frames:
                        lesion_candidates.append({"start": max(0, abs_idx - args.consecutive_frames + 1), "avg_conf": current_conf_sum / args.consecutive_frames})
                else:
                    consecutive_hits, current_conf_sum = 0, 0.0

        # 筛选与保存
        target_clips = []
        lesion_candidates.sort(key=lambda x: x["avg_conf"], reverse=True)
        lesion_starts = []
        lesion_clip_count = 0
        for cand in lesion_candidates:
            if len(target_clips) >= 6: break
            cand_start = max(0, min(cand["start"], max(0, total_frames - 60)))
            if any(abs(cand_start - s) < 60 for s in lesion_starts): continue
            target_clips.append({"start": cand_start, "type": "Lesion"})
            lesion_starts.append(cand_start)
            lesion_clip_count += 1

        if len(target_clips) < 6:
            needed = 6 - len(target_clips)
            if total_frames > 60:
                step = total_frames // (needed + 1)
                for k in range(1, needed + 1):
                    fill_start = max(0, min(k * step, total_frames - 60))
                    if any(abs(fill_start - s) < 60 for s in lesion_starts):
                        found = False
                        for offset in range(60, total_frames, 60):
                            new_fill = (fill_start + offset) % (total_frames - 60)
                            if not any(abs(new_fill - s) < 60 for s in lesion_starts):
                                fill_start = new_fill
                                found = True
                                break
                        if not found:
                            fill_start = max(0, min(fill_start, total_frames - 60))
                    target_clips.append({"start": fill_start, "type": "NonLesion"})
            else:
                for _ in range(needed):
                    target_clips.append({"start": 0, "type": "NonLesion"})

        while len(target_clips) < 6:
            target_clips.append({"start": 0, "type": "NonLesion"})
        target_clips = target_clips[:6]

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        for c_idx, clip_info in enumerate(target_clips):
            start_f = clip_info["start"]
            clip_path = os.path.join(clip_subfolder, f"{clip_info['type']}_clip_{c_idx+1:02d}_start_{start_f}.mp4")
            out_writer = cv2.VideoWriter(clip_path, fourcc, fps, (width, height))
            for f_offset in range(60):
                read_idx = start_f + f_offset
                out_writer.write(all_frames[read_idx] if read_idx < total_frames else all_frames[-1])
            out_writer.release()

        worker_results[filename] = video_frame_results
        worker_lesion_clip_stats[video_path] = lesion_clip_count # 使用全路径作为Key以便Master解析文件夹
        print(f"[Worker {worker_id}] {filename} processed, Lesion: {lesion_clip_count}", flush=True)
        gc.collect()
        torch.cuda.empty_cache()

    with open(f"temp_result_worker_{worker_id}.json", 'w') as f: json.dump(worker_results, f)
    with open(f"temp_stats_worker_{worker_id}.json", 'w') as f: json.dump(worker_lesion_clip_stats, f)

# ================= 5. Master 逻辑 =================
def run_master(args):
    target_class = args.target_class
    print("="*60)
    print(f"Target Class: {target_class}")
    if os.path.exists(args.clip_save_dir): shutil.rmtree(args.clip_save_dir)
    os.makedirs(args.clip_save_dir, exist_ok=True)

    video_paths = []
    exts = {'.mp4', '.wmv', '.avi', '.mov', '.mkv', '.MP4', '.WMV', '.AVI'}
    scan_dir = args.input_folder if target_class == "all" else os.path.join(args.input_folder, target_class)
    for root, _, files in os.walk(scan_dir):
        for f in files:
            if any(f.endswith(e) for e in exts): video_paths.append(os.path.join(root, f))

    if not video_paths: return
    random.shuffle(video_paths)
    total_workers = args.num_gpus * args.workers_per_gpu
    chunks = np.array_split(video_paths, total_workers)
    processes = []

    for i in range(total_workers):
        if len(chunks[i]) == 0: continue
        chunk_file = f"temp_chunk_dual_{i}.json"
        with open(chunk_file, 'w') as f: json.dump(chunks[i].tolist(), f)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(i // args.workers_per_gpu)
        cmd = [
            sys.executable, os.path.abspath(__file__),
            "--worker_mode", "--worker_id", str(i), "--chunk_json", chunk_file,
            "--input_folder", args.input_folder,
            "--output_json_name", args.output_json_name,
            "--yolo_model_path", args.yolo_model_path,
            "--svm_model_path", args.svm_model_path,
            "--num_gpus", str(args.num_gpus),
            "--workers_per_gpu", str(args.workers_per_gpu),
            "--batch_size", str(args.batch_size),
            "--yolo_img_size", str(args.yolo_img_size),
            "--yolo_exclude_class", str(args.yolo_exclude_class),
            "--yolo_conf_threshold", str(args.yolo_conf_threshold),
            "--consecutive_frames", str(args.consecutive_frames),
            "--clip_save_dir", args.clip_save_dir,
        ]
        processes.append(subprocess.Popen(cmd, env=env))
    for p in processes: p.wait()

    final_results = {}
    all_lesion_stats = {} # 汇总所有视频的路径和病灶数量
    for i in range(total_workers):
        res_file, stats_file = f"temp_result_worker_{i}.json", f"temp_stats_worker_{i}.json"
        if os.path.exists(res_file):
            with open(res_file, 'r') as f: final_results.update(json.load(f))
            os.remove(res_file)
        if os.path.exists(stats_file):
            with open(stats_file, 'r') as f: all_lesion_stats.update(json.load(f))
            os.remove(stats_file)
        if os.path.exists(f"temp_chunk_dual_{i}.json"): os.remove(f"temp_chunk_dual_{i}.json")

    with open(args.output_json_name, 'w', encoding='utf-8') as f: json.dump(final_results, f, indent=4)

    # --- 核心修改：按文件夹统计比例 ---
    folder_report = {} # {folder_name: {'total': 0, 'lesion': 0}}
    for v_path, count in all_lesion_stats.items():
        rel_path = os.path.relpath(v_path, args.input_folder)
        folder_name = rel_path.split(os.sep)[0] # 获取 class_0, class_1 等
        if folder_name not in folder_report:
            folder_report[folder_name] = {'total': 0, 'lesion': 0}
        folder_report[folder_name]['total'] += 1
        if count > 0:
            folder_report[folder_name]['lesion'] += 1

    print("\n" + "="*60)
    print("【Folder-wise Lesion Video Ratio Report】")
    total_all, lesion_all = 0, 0
    for folder, data in sorted(folder_report.items()):
        ratio = (data['lesion'] / data['total'] * 100) if data['total'] > 0 else 0
        print(f"Folder [{folder}]: {data['lesion']}/{data['total']} videos contains Lesion Clips ({ratio:.2f}%)")
        total_all += data['total']
        lesion_all += data['lesion']

    total_ratio = (lesion_all / total_all * 100) if total_all > 0 else 0
    print("-" * 30)
    print(f"Overall Summary: {lesion_all}/{total_all} videos ({total_ratio:.2f}%)")
    print("="*60 + "\n")

if __name__ == "__main__":
    args = parse_args()

    if args.worker_mode:
        run_worker(args.chunk_json, args.worker_id, args)
    else:
        run_master(args)
