from huggingface_hub import snapshot_download
import argparse
import os
import sys

os.environ["HF_HUB_DISABLE_XET"] = "1"

REPO_ID = "xenosscu/BUVFM"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download datasets and/or checkpoints from HuggingFace.")
    parser.add_argument(
        "--what",
        choices=["both", "dataset", "ckpt"],
        default="both",
        help="What to download: 'dataset' only, 'ckpt' only, or 'both' (default).",
    )
    parser.add_argument("--dataset_dir", default="./dataset", help="Output directory for dataset (default: ./dataset)")
    parser.add_argument("--model_dir", default=".", help="Output directory for checkpoints (default: .)")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("Error: HF_TOKEN environment variable not set.")
        sys.exit(1)

    targets = []
    if args.what in ("both", "dataset"):
        targets.append(("dataset", args.dataset_dir))
    if args.what in ("both", "ckpt"):
        targets.append(("model", args.model_dir))

    for repo_type, output_dir in targets:
        print(f"Downloading {REPO_ID} (type={repo_type}) to '{output_dir}' ...")
        snapshot_download(
            repo_id=REPO_ID,
            repo_type=repo_type,
            local_dir=output_dir,
            local_dir_use_symlinks=False,
            token=token,
        )
        print(f"Done ({repo_type}).")

    print("All downloads complete.")
