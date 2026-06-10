from huggingface_hub import snapshot_download
import os
import sys

os.environ["HF_HUB_DISABLE_XET"] = "1"

REPO_ID = "xenosscu/BUVFM"

if __name__ == "__main__":
    if "--help" in sys.argv:
        print("Usage: python download_dataset.py [dataset_dir [model_dir]]")
        print("  dataset_dir: default ./dataset")
        print("  model_dir:   default .")
        sys.exit(0)

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("Error: HF_TOKEN environment variable not set.")
        sys.exit(1)

    targets = [
        ("dataset", sys.argv[1] if len(sys.argv) >= 2 else "./dataset"),
        ("model",   sys.argv[2] if len(sys.argv) >= 3 else "."),
    ]

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
