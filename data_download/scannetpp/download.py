from huggingface_hub import snapshot_download
import os

# Configuration
repo_id = "lifuguan/InsScene-15K"
# Target the entire processed_scannetpp_v2 directory. Files are split-volume
# archives (processed_scannetpp_v2.zip.001, .002, ...) — keep all of them.
folder_pattern = "processed_scannetpp_v2/*"

# ── Root path ────────────────────────────────────────────────────────────────
# Set SCANNETPP_ROOT to control where downloaded data is stored.
# Priority: env var > DEFAULT_ROOT below.
DEFAULT_ROOT = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/native_mm/zhangquan/code/lhxk/workspace/streamIGGT/data"  # <-- change this for your setup
local_dir = os.environ.get("SCANNETPP_ROOT", DEFAULT_ROOT)
# ─────────────────────────────────────────────────────────────────────────────

# Ensure the directory exists
os.makedirs(local_dir, exist_ok=True)

print(f"Starting download of all shards in {repo_id}/processed_scannetpp_v2...")

try:
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=folder_pattern,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        # Adding resume_download to handle potential network interruptions in restricted environments
        resume_download=True
    )
    print(f"Download complete! Files are located at: {local_dir}")
except Exception as e:
    print(f"An error occurred: {e}")
