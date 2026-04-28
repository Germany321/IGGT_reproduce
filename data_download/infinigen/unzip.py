import os
import zipfile
from tqdm import tqdm

# ── Root paths ───────────────────────────────────────────────────────────────
# Set INFINIGEN_ROOT to the directory that contains `processed_infinigen/`
# (i.e. the same directory you passed as `local_dir` in download.py).
# Set INFINIGEN_EXTRACT_ROOT to where extracted scenes should be written.
# Priority for both: env var > DEFAULT_* below.
DEFAULT_SOURCE_ROOT = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/native_mm/zhangquan/code/lhxk/workspace/streamIGGT/data/processed_infinigen"
DEFAULT_EXTRACT_ROOT = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/native_mm/zhangquan/code/lhxk/workspace/streamIGGT/data/infinigen_extracted"

SOURCE_ROOT = os.environ.get("INFINIGEN_SOURCE_ROOT", DEFAULT_SOURCE_ROOT)
EXTRACT_ROOT = os.environ.get("INFINIGEN_EXTRACT_ROOT", DEFAULT_EXTRACT_ROOT)
# ─────────────────────────────────────────────────────────────────────────────


def unzip_per_scene(source_root, extract_to_root):
    """
    For each scene_XXX folder in source_root, extract all of its zip files
    into extract_to_root/scene_XXX/. Each zip inside a scene is treated as
    one sub-asset of that scene (e.g. one camera/view) and is extracted into
    its own subfolder named after the zip stem so files from different zips
    don't collide.

    Layout produced:
        extract_to_root/
            scene_000/
                <zip_stem_a>/...
                <zip_stem_b>/...
            scene_001/
                ...
    """
    if not os.path.isdir(source_root):
        raise FileNotFoundError(f"SOURCE_ROOT does not exist: {source_root}")

    scene_dirs = sorted(
        d for d in os.listdir(source_root)
        if d.startswith("scene_") and os.path.isdir(os.path.join(source_root, d))
    )
    print(f"Found {len(scene_dirs)} scene folders under {source_root}.")

    total_zips = 0
    for scene in scene_dirs:
        scene_src = os.path.join(source_root, scene)
        scene_dst = os.path.join(extract_to_root, scene)
        os.makedirs(scene_dst, exist_ok=True)

        zips = sorted(f for f in os.listdir(scene_src) if f.endswith(".zip"))
        total_zips += len(zips)

        for zname in tqdm(zips, desc=f"Extracting {scene}", leave=False):
            zip_path = os.path.join(scene_src, zname)
            stem = os.path.splitext(zname)[0]
            out_dir = os.path.join(scene_dst, stem)
            os.makedirs(out_dir, exist_ok=True)
            try:
                with zipfile.ZipFile(zip_path, "r") as zf:
                    zf.extractall(out_dir)
            except zipfile.BadZipFile:
                print(f"Skipping {zip_path}: not a valid zip file.")
            except Exception as e:
                print(f"Error extracting {zip_path}: {e}")

    print(f"\nExtraction complete! {total_zips} archives across {len(scene_dirs)} scenes.")


if __name__ == "__main__":
    os.makedirs(EXTRACT_ROOT, exist_ok=True)
    unzip_per_scene(SOURCE_ROOT, EXTRACT_ROOT)
    print(f"Output: {EXTRACT_ROOT}")
