#!/usr/bin/env python3
"""Generate annotation files for VideoMAEv2 pretraining from .mp4 directories.

Usage:
    python pretraining/prepare_data.py \
        --video_dirs /home/wcz/workspace/DATASET/us_foundation_model_dataset_videos_videos \
                     /home/wcz/workspace/DATASET/us_foundation_model_dataset_img_videos \
        --data_root /home/wcz/workspace/DATASET \
        --output_dir pretraining/data
"""

import argparse
import glob
import os


def collect_videos(video_dirs, data_root):
    """Collect all .mp4 files, return relative paths from data_root."""
    videos = []
    for d in video_dirs:
        mp4_files = sorted(glob.glob(os.path.join(d, "*.mp4")))
        for f in mp4_files:
            rel_path = os.path.relpath(f, data_root)
            videos.append(rel_path)
    return videos


def write_videomae_format(videos, output_path):
    """VideoMAEv2 format: video_path start_idx total_frame
    total_frame < 0 signals decord-based loading from .mp4 file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        for v in videos:
            f.write(f"{v} 0 -1\n")
    print(f"[VideoMAEv2] Wrote {len(videos)} videos to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare pretraining data files")
    parser.add_argument("--video_dirs", nargs="+", required=True,
                        help="Directories containing .mp4 files")
    parser.add_argument("--data_root", required=True,
                        help="Root directory for relative paths in annotation files")
    parser.add_argument("--output_dir", default="pretraining/data",
                        help="Output directory for annotation files")
    args = parser.parse_args()

    videos = collect_videos(args.video_dirs, args.data_root)
    print(f"Found {len(videos)} .mp4 files total")

    if len(videos) == 0:
        print("WARNING: No .mp4 files found!")
        return

    videomae_path = os.path.join(args.output_dir, "us_videomae_train.txt")
    write_videomae_format(videos, videomae_path)


if __name__ == "__main__":
    main()
