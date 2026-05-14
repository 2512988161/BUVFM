from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="xenosscu/BUVFM", 
    repo_type="model",
    local_dir="./",
    local_dir_use_symlinks=False,
)  