"""
Download a pretrained VGGT/IGGT checkpoint from ModelScope and write it as a
plain torch ``.pt`` state_dict file that the trainer's
``checkpoint.resume_checkpoint_path`` can consume.

ModelScope is used here instead of Hugging Face because the HF endpoint is
unreliable / blocked in some networks (the HF download was being killed).
ModelScope mirrors most popular models and is reachable from CN networks
without a proxy.

The trainer loads the produced ``.pt`` with ``strict=False`` so the
IGGT-only modules (``part_adaptor``, ``part_head``, the extended
``point_head``) stay randomly initialised while the aggregator and other
heads inherit pretrained weights.

Usage:
    # Use the defaults set below
    python data_download/prepare_pretrained.py

    # CLI overrides
    python data_download/prepare_pretrained.py \
        --ms-repo OpenGVLab/VGGT-1B \
        --out-dir /abs/path/to/checkpoints \
        --filename vggt_1b.pt

    # Env-var overrides (handy for Docker / CI)
    MS_REPO=OpenGVLab/VGGT-1B \
    PRETRAINED_DIR=/abs/path/to/checkpoints \
    PRETRAINED_FILENAME=vggt_1b.pt \
    python data_download/prepare_pretrained.py
"""

import argparse
import glob
import os
import os.path as osp
import sys

import torch


# ── Defaults ─────────────────────────────────────────────────────────────────
# Edit these for a permanent change, or use --out-dir / PRETRAINED_DIR per run.
# The ModelScope repo id needs to host a VGGT-1B-compatible checkpoint
# (single model.safetensors or pytorch_model.bin with the same key layout as
# facebook/VGGT-1B). If your org hosts a private mirror, set MS_REPO to it.
DEFAULT_MS_REPO = "OpenGVLab/VGGT-1B"
DEFAULT_PRETRAINED_DIR = osp.abspath(
    osp.join(osp.dirname(osp.abspath(__file__)), "..", "checkpoints")
)
DEFAULT_FILENAME = "vggt_1b.pt"
# ─────────────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--ms-repo",
        default=os.environ.get("MS_REPO", DEFAULT_MS_REPO),
        help=f"ModelScope model id to pull (default: {DEFAULT_MS_REPO}).",
    )
    p.add_argument(
        "--out-dir",
        default=os.environ.get("PRETRAINED_DIR", DEFAULT_PRETRAINED_DIR),
        help=f"Directory to write the .pt file into (default: {DEFAULT_PRETRAINED_DIR}).",
    )
    p.add_argument(
        "--filename",
        default=os.environ.get("PRETRAINED_FILENAME", DEFAULT_FILENAME),
        help=f"Output filename inside --out-dir (default: {DEFAULT_FILENAME}).",
    )
    p.add_argument(
        "--cache-dir",
        default=os.environ.get("MODELSCOPE_CACHE", None),
        help="ModelScope cache dir (defaults to ~/.cache/modelscope).",
    )
    return p.parse_args()


def _load_state_dict_from_dir(model_dir: str) -> dict:
    """Locate weight shards in a ModelScope snapshot dir and merge them.

    Supports both ``*.safetensors`` (preferred) and ``pytorch_model*.bin``
    layouts. Sharded checkpoints are stitched together by concatenating
    every shard's state_dict — keys are unique across shards by HF
    convention so this is safe.
    """
    safetensor_files = sorted(glob.glob(osp.join(model_dir, "*.safetensors")))
    if safetensor_files:
        try:
            from safetensors.torch import load_file
        except ImportError as e:
            raise ImportError(
                "safetensors is required to load .safetensors checkpoints. "
                "Install with `pip install safetensors`."
            ) from e
        state_dict = {}
        for f in safetensor_files:
            print(f"  loading {osp.basename(f)} ...")
            state_dict.update(load_file(f, device="cpu"))
        return state_dict

    bin_files = sorted(glob.glob(osp.join(model_dir, "pytorch_model*.bin")))
    if bin_files:
        state_dict = {}
        for f in bin_files:
            print(f"  loading {osp.basename(f)} ...")
            shard = torch.load(f, map_location="cpu")
            # HF index files wrap shards under {"state_dict": ...} sometimes
            if isinstance(shard, dict) and "state_dict" in shard and isinstance(shard["state_dict"], dict):
                shard = shard["state_dict"]
            state_dict.update(shard)
        return state_dict

    raise FileNotFoundError(
        f"No .safetensors or pytorch_model*.bin files found under {model_dir}. "
        f"Contents: {os.listdir(model_dir)}"
    )


def main():
    args = parse_args()
    out_path = osp.abspath(osp.join(args.out_dir, args.filename))
    os.makedirs(osp.dirname(out_path), exist_ok=True)

    try:
        from modelscope import snapshot_download
    except ImportError:
        print(
            "ERROR: modelscope is not installed. Install it with:\n"
            "    pip install modelscope\n",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Downloading {args.ms_repo} from ModelScope ...")
    model_dir = snapshot_download(
        args.ms_repo,
        cache_dir=args.cache_dir,
    )
    print(f"Snapshot at: {model_dir}")

    print("Reading weight shards ...")
    state_dict = _load_state_dict_from_dir(model_dir)

    n_params = sum(v.numel() for v in state_dict.values()) / 1e6
    print(f"Saving state_dict ({n_params:.1f}M params, {len(state_dict)} keys) -> {out_path}")
    # Wrap in {"model": ...} so the trainer's _load_resuming_checkpoint picks
    # it up correctly. Optimizer/epoch/scaler are intentionally omitted so
    # resuming from this file behaves as "load weights only, fresh optimizer".
    torch.save({"model": state_dict}, out_path)
    print("Done.")
    print(f"\nNext: set checkpoint.resume_checkpoint_path: {out_path} in your config.")


if __name__ == "__main__":
    main()
