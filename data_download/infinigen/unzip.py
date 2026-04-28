import os
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
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

# Tune via env: INFINIGEN_NUM_WORKERS (defaults to os.cpu_count()).
NUM_WORKERS = int(os.environ.get("INFINIGEN_NUM_WORKERS", os.cpu_count() or 8))
# ─────────────────────────────────────────────────────────────────────────────


def _extract_one(task):
    """Worker: extract a single zip into its own subfolder.

    Returns (zip_path, ok, err_msg). Used by ProcessPoolExecutor so it must
    be a top-level picklable function.
    """
    zip_path, out_dir = task
    try:
        os.makedirs(out_dir, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(out_dir)
        return zip_path, True, None
    except zipfile.BadZipFile:
        return zip_path, False, "not a valid zip file"
    except Exception as e:
        return zip_path, False, repr(e)


def _build_tasks(source_root, extract_to_root):
    if not os.path.isdir(source_root):
        raise FileNotFoundError(f"SOURCE_ROOT does not exist: {source_root}")

    scene_dirs = sorted(
        d for d in os.listdir(source_root)
        if d.startswith("scene_") and os.path.isdir(os.path.join(source_root, d))
    )

    tasks = []
    for scene in scene_dirs:
        scene_src = os.path.join(source_root, scene)
        scene_dst = os.path.join(extract_to_root, scene)
        os.makedirs(scene_dst, exist_ok=True)
        for zname in sorted(f for f in os.listdir(scene_src) if f.endswith(".zip")):
            zip_path = os.path.join(scene_src, zname)
            stem = os.path.splitext(zname)[0]
            out_dir = os.path.join(scene_dst, stem)
            tasks.append((zip_path, out_dir))
    return scene_dirs, tasks


def unzip_per_scene(source_root, extract_to_root, num_workers=NUM_WORKERS):
    """
    For each scene_XXX folder in source_root, extract all of its zip files
    into extract_to_root/scene_XXX/<zip_stem>/ in parallel using a process
    pool. Per-zip granularity gives finer load balancing than per-scene.

    Layout produced:
        extract_to_root/
            scene_000/
                <zip_stem_a>/...
                <zip_stem_b>/...
            scene_001/
                ...
    """
    scene_dirs, tasks = _build_tasks(source_root, extract_to_root)
    print(
        f"Found {len(scene_dirs)} scene folders, {len(tasks)} archives total. "
        f"Extracting with {num_workers} workers."
    )

    if not tasks:
        return

    # chunksize trades off scheduling overhead vs. tail latency; small here
    # because individual zips can vary a lot in size.
    failures = []
    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        futures = [pool.submit(_extract_one, t) for t in tasks]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Extracting"):
            zip_path, ok, err = fut.result()
            if not ok:
                failures.append((zip_path, err))

    if failures:
        print(f"\n{len(failures)} archive(s) failed:")
        for p, err in failures[:20]:
            print(f"  {p}: {err}")
        if len(failures) > 20:
            print(f"  ... and {len(failures) - 20} more")
    print(
        f"\nExtraction complete! {len(tasks) - len(failures)}/{len(tasks)} "
        f"archives across {len(scene_dirs)} scenes."
    )


if __name__ == "__main__":
    os.makedirs(EXTRACT_ROOT, exist_ok=True)
    unzip_per_scene(SOURCE_ROOT, EXTRACT_ROOT)
    print(f"Output: {EXTRACT_ROOT}")
