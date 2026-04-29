"""
Reassemble and extract the ScanNet++ split-volume archive shipped under
``processed_scannetpp_v2/`` in the InsScene-15K HF dataset.

Unlike infinigen (which ships one zip per scene), ScanNet++ here is a single
large logical archive split into ~4 GB shards named ``*.zip.001``, ``.002``,
…, ``.NNN``. We:

  1. Concatenate the shards in order into one combined ``.zip`` (streamed,
     no double-buffering — the combined file is the same size as the sum of
     shards, ~227 GB, so make sure the destination has enough free space).
  2. Extract the combined zip into ``EXTRACT_ROOT``.
  3. Optionally delete the combined zip when done (keep the shards as the
     authoritative copy).

The reassemble step is parallelism-resistant — it must be a single ordered
concatenation. Extraction itself is parallelised across files inside the
combined zip.
"""

import argparse
import os
import os.path as osp
import re
import zipfile
from tqdm import tqdm


# ── Root paths ───────────────────────────────────────────────────────────────
# SOURCE_ROOT contains the .zip.NNN shards (= local_dir from download.py
# joined with `processed_scannetpp_v2`).
# EXTRACT_ROOT is where the unpacked dataset will live.
DEFAULT_SOURCE_ROOT = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/native_mm/zhangquan/code/lhxk/workspace/streamIGGT/data/processed_scannetpp_v2"
DEFAULT_EXTRACT_ROOT = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/native_mm/zhangquan/code/lhxk/workspace/streamIGGT/data/processed_scannetpp_v2_extracted"

SOURCE_ROOT = os.environ.get("SCANNETPP_SOURCE_ROOT", DEFAULT_SOURCE_ROOT)
EXTRACT_ROOT = os.environ.get("SCANNETPP_EXTRACT_ROOT", DEFAULT_EXTRACT_ROOT)

# Where to write the reassembled zip. Defaults to inside SOURCE_ROOT so it
# sits next to the shards. Override with --combined-path / SCANNETPP_COMBINED_ZIP
# if you want it on a different disk.
DEFAULT_COMBINED_NAME = "processed_scannetpp_v2.combined.zip"
# ─────────────────────────────────────────────────────────────────────────────


_SHARD_RE = re.compile(r".*\.zip\.(\d+)$")


def _list_shards(source_root: str):
    if not osp.isdir(source_root):
        raise FileNotFoundError(f"SOURCE_ROOT does not exist: {source_root}")
    shards = []
    for f in os.listdir(source_root):
        m = _SHARD_RE.match(f)
        if m:
            shards.append((int(m.group(1)), osp.join(source_root, f)))
    if not shards:
        raise FileNotFoundError(
            f"No .zip.NNN shards found under {source_root}. "
            f"Did the download finish?"
        )
    shards.sort(key=lambda x: x[0])

    # Sanity check: shard indices should be contiguous starting at 1.
    expected = list(range(1, len(shards) + 1))
    actual = [i for i, _ in shards]
    if actual != expected:
        missing = sorted(set(expected) - set(actual))
        raise RuntimeError(
            f"Shard indices not contiguous. Missing: {missing[:10]}"
            f"{' ...' if len(missing) > 10 else ''}"
        )
    return [p for _, p in shards]


def reassemble(source_root: str, combined_path: str, force: bool = False) -> str:
    """Concatenate every .zip.NNN shard in source_root into combined_path."""
    if osp.exists(combined_path) and not force:
        size = osp.getsize(combined_path)
        print(
            f"Combined zip already exists at {combined_path} "
            f"({size / 1e9:.1f} GB). Pass --force to rebuild."
        )
        return combined_path

    shards = _list_shards(source_root)
    total_bytes = sum(osp.getsize(p) for p in shards)
    print(
        f"Reassembling {len(shards)} shards "
        f"({total_bytes / 1e9:.1f} GB) -> {combined_path}"
    )

    os.makedirs(osp.dirname(combined_path) or ".", exist_ok=True)
    chunk = 8 * 1024 * 1024  # 8 MiB
    with open(combined_path, "wb") as out, tqdm(
        total=total_bytes, unit="B", unit_scale=True, desc="Joining shards"
    ) as bar:
        for shard in shards:
            with open(shard, "rb") as inp:
                while True:
                    buf = inp.read(chunk)
                    if not buf:
                        break
                    out.write(buf)
                    bar.update(len(buf))
    return combined_path


def extract(combined_path: str, extract_to_root: str):
    """Extract every member of the combined zip into extract_to_root.

    Single-process by design: with 3M+ small files on a networked filesystem
    the bottleneck is FS metadata ops, not CPU, so multi-process parallelism
    barely helps and risks OOM-killing workers (each fork carries a full
    central-directory copy in memory, ~600 MB for a 3M-entry zip).

    For real parallel extraction of an archive this size, prefer the `7z` CLI
    (`7z x combined.zip -mmt=16 -o<dst>`); see README.
    """
    os.makedirs(extract_to_root, exist_ok=True)

    print(f"Opening {combined_path} (this scans the central directory) ...")
    with zipfile.ZipFile(combined_path, "r") as zf:
        members = [info for info in zf.infolist() if not info.is_dir()]
        print(f"Extracting {len(members)} files -> {extract_to_root}")
        if not members:
            print("No files to extract.")
            return

        failures = []
        # tqdm.update on every file keeps the bar smooth even when individual
        # extracts are sub-millisecond on the local cache hits.
        for info in tqdm(members, desc="Extracting", unit="file"):
            try:
                zf.extract(info, extract_to_root)
            except Exception as e:
                failures.append((info.filename, repr(e)))

    if failures:
        print(f"\n{len(failures)} file(s) failed:")
        for name, err in failures[:20]:
            print(f"  {name}: {err}")
        if len(failures) > 20:
            print(f"  ... and {len(failures) - 20} more")
    print(f"\nExtraction complete! {len(members) - len(failures)}/{len(members)} files.")


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--source-root", default=SOURCE_ROOT, help="Dir containing .zip.NNN shards.")
    p.add_argument("--extract-root", default=EXTRACT_ROOT, help="Dir to extract into.")
    p.add_argument(
        "--combined-path",
        default=os.environ.get(
            "SCANNETPP_COMBINED_ZIP",
            osp.join(SOURCE_ROOT, DEFAULT_COMBINED_NAME),
        ),
        help="Where to write the reassembled .zip (defaults inside source root).",
    )
    p.add_argument("--force", action="store_true", help="Re-reassemble even if combined zip exists.")
    p.add_argument("--keep-combined", action="store_true",
                   help="Don't delete the combined zip after a successful extract.")
    p.add_argument("--reassemble-only", action="store_true",
                   help="Stop after reassembly (skip extraction).")
    p.add_argument("--extract-only", action="store_true",
                   help="Skip reassembly; assume --combined-path already exists.")
    return p.parse_args()


def main():
    args = parse_args()

    if not args.extract_only:
        reassemble(args.source_root, args.combined_path, force=args.force)
    if args.reassemble_only:
        print(f"Reassembly only — combined zip at: {args.combined_path}")
        return

    extract(args.combined_path, args.extract_root)

    if not args.keep_combined:
        try:
            os.remove(args.combined_path)
            print(f"Removed intermediate combined zip: {args.combined_path}")
        except OSError as e:
            print(f"Could not remove combined zip: {e}")

    print(f"Output: {args.extract_root}")


if __name__ == "__main__":
    main()
