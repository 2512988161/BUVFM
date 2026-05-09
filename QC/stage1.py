import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import timm
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
cv2.setNumThreads(0)

YOLO_MODEL_PATH = "./best_640_s_60e(2).pt"
SVM_MODEL_PATH = "./mobilenetv3_small_075_yl_241222(3).pth"
YOLO_IMG_SIZE = 256
YOLO_EXCLUDE_CLASS = 1
YOLO_CONF_THRESHOLD = 0.5
BATCH_SIZE = 16
VIDEO_EXTS = {".mp4", ".wmv", ".avi", ".mov", ".mkv", ".MP4", ".WMV", ".AVI", ".MOV", ".MKV"}
OUTPUT_DIR = Path("./output")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=str, help="mp4 file path or directory path")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    return parser.parse_args()


def collect_video_paths(input_path: Path):
    if input_path.is_file():
        if input_path.suffix in VIDEO_EXTS:
            return [input_path]
        raise ValueError(f"Unsupported video file: {input_path}")

    if input_path.is_dir():
        video_paths = [p for p in input_path.rglob("*") if p.is_file() and p.suffix in VIDEO_EXTS]
        return sorted(video_paths)

    raise FileNotFoundError(f"Input path not found: {input_path}")


def resolve_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_mobilenet(device):
    model = timm.create_model("mobilenetv3_small_075", pretrained=False, in_chans=3, num_classes=2).to(device)
    if not os.path.exists(SVM_MODEL_PATH):
        raise FileNotFoundError(f"MobileNet checkpoint not found: {SVM_MODEL_PATH}")
    model.load_state_dict(torch.load(SVM_MODEL_PATH, map_location=device))
    model.eval()
    return model


def build_yolo(device):
    from ultralytics import YOLO

    model = YOLO(YOLO_MODEL_PATH)
    model(np.zeros((YOLO_IMG_SIZE, YOLO_IMG_SIZE, 3), dtype=np.uint8), device=str(device), verbose=False)
    return model


def load_models(device):
    svm_model = build_mobilenet(device)
    try:
        yolo_model = build_yolo(device)
        yolo_device = device
    except torch.AcceleratorError as e:
        if device.type != "cuda":
            raise
        print(f"YOLO CUDA init failed, fallback to CPU: {e}")
        yolo_model = build_yolo(torch.device("cpu"))
        yolo_device = torch.device("cpu")
    return svm_model, yolo_model, yolo_device


def build_preprocess():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])


def draw_mobilenet_score(frame, score):
    valid = 1.0 - score
    text = f"Valid Frame: {valid:.3f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.58
    thickness = 2
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x = max(10, frame.shape[1] - tw - 20)
    y = 24 + th
    cv2.putText(frame, text, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def draw_yolo_boxes(frame, boxes):
    color = (0, 255, 0)
    for box in boxes:
        x1, y1, x2, y2 = map(int, box["xyxy"])
        conf = float(box["conf"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"Conf:{conf:.3f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.6
        thickness = 2
        (tw, th), baseline = cv2.getTextSize(label, font, scale, thickness)
        ty = max(th + baseline + 6, y1)
        cv2.putText(frame, label, (x1 + 3, ty - baseline - 2), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def infer_video(video_path: Path, svm_model, yolo_model, preprocess, svm_device, yolo_device, batch_size: int, video_output_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_output_path), fourcc, fps, (width, height))

    frame_results = []
    batch_frames = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        batch_frames.append(frame)
        if len(batch_frames) == batch_size:
            batch_results, annotated_frames = infer_batch(
                batch_frames,
                frame_idx - len(batch_frames) + 1,
                svm_model,
                yolo_model,
                preprocess,
                svm_device,
                yolo_device,
            )
            frame_results.extend(batch_results)
            for annotated in annotated_frames:
                writer.write(annotated)
            batch_frames = []
        frame_idx += 1

    if batch_frames:
        batch_results, annotated_frames = infer_batch(
            batch_frames,
            frame_idx - len(batch_frames),
            svm_model,
            yolo_model,
            preprocess,
            svm_device,
            yolo_device,
        )
        frame_results.extend(batch_results)
        for annotated in annotated_frames:
            writer.write(annotated)

    cap.release()
    writer.release()
    return frame_results


def infer_batch(batch_frames, start_idx, svm_model, yolo_model, preprocess, svm_device, yolo_device):
    imgs_pil = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in batch_frames]
    svm_tensors = torch.stack([preprocess(img) for img in imgs_pil]).to(svm_device)

    with torch.inference_mode():
        logits = svm_model(svm_tensors)
        probs = F.softmax(logits, dim=1)
        mb_scores = probs[:, 1].cpu().tolist()

    yolo_scores = [0.0] * len(batch_frames)
    yolo_boxes = [[] for _ in batch_frames]
    y_indices = [idx for idx, score in enumerate(mb_scores) if score < 0.5]
    if y_indices:
        y_inputs = [batch_frames[idx] for idx in y_indices]
        y_results = yolo_model(
            y_inputs,
            imgsz=YOLO_IMG_SIZE,
            conf=YOLO_CONF_THRESHOLD,
            device=str(yolo_device),
            verbose=False,
        )
        for res_idx, batch_idx in enumerate(y_indices):
            res = y_results[res_idx]
            if res.boxes is None:
                continue
            frame_boxes = []
            for box in res.boxes:
                if int(box.cls[0]) == YOLO_EXCLUDE_CLASS:
                    continue
                conf = float(box.conf[0])
                xyxy = [float(v) for v in box.xyxy[0].tolist()]
                frame_boxes.append({
                    "xyxy": xyxy,
                    "conf": conf,
                    "cls": int(box.cls[0]),
                })
            yolo_boxes[batch_idx] = frame_boxes
            yolo_scores[batch_idx] = max((box["conf"] for box in frame_boxes), default=0.0)

    batch_results = []
    annotated_frames = []
    for offset, frame in enumerate(batch_frames):
        mobilenet_score = mb_scores[offset]
        yolo_score = yolo_scores[offset]
        boxes = yolo_boxes[offset]

        annotated = frame.copy()
        draw_yolo_boxes(annotated, boxes)
        draw_mobilenet_score(annotated, mobilenet_score)
        annotated_frames.append(annotated)

        batch_results.append({
            "frame_idx": start_idx + offset,
            "yolo_score": round(yolo_score, 4),
            "mobilenet_score": round(mobilenet_score, 4),
            "yolo_boxes": [
                {
                    "xyxy": [round(v, 2) for v in box["xyxy"]],
                    "conf": round(box["conf"], 4),
                    "cls": box["cls"],
                }
                for box in boxes
            ],
        })
    return batch_results, annotated_frames


def make_output_paths(video_path: Path):
    stem = video_path.stem
    return OUTPUT_DIR / f"{stem}.json", OUTPUT_DIR / f"{stem}.mp4"


def main():
    args = parse_args()
    input_path = Path(args.input_path)
    video_paths = collect_video_paths(input_path)
    if not video_paths:
        print("No videos found.")
        return

    OUTPUT_DIR.mkdir(exist_ok=True)

    device = resolve_device()
    print(f"Loading models with MobileNet on {device}...")
    svm_model, yolo_model, yolo_device = load_models(device)
    print(f"YOLO device: {yolo_device}")
    preprocess = build_preprocess()

    for video_path in video_paths:
        print(f"Processing {video_path}")
        json_output_path, video_output_path = make_output_paths(video_path)
        frame_results = infer_video(
            video_path,
            svm_model,
            yolo_model,
            preprocess,
            device,
            yolo_device,
            args.batch_size,
            video_output_path,
        )
        with open(json_output_path, "w", encoding="utf-8") as f:
            json.dump(frame_results, f, ensure_ascii=False, indent=2)
        print(f"Saved {json_output_path}")
        print(f"Saved {video_output_path}")


if __name__ == "__main__":
    main()
