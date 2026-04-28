"""
Download a pretrained VGGT/IGGT checkpoint from Hugging Face and write it as
a plain torch ``.pt`` state_dict file that the trainer's
``checkpoint.resume_checkpoint_path`` can consume.

Default: pulls ``facebook/VGGT-1B`` (the backbone IGGT fine-tunes from). The
trainer loads this with ``strict=False`` so the IGGT-only modules
(``part_adaptor``, ``part_head``, the extended ``point_head``) stay randomly
initialized while the aggregator and other heads inherit pretrained weights.

Usage:
    # Use the defaults set below
    python data_download/prepare_pretrained.py

    # CLI overrides
    python data_download/prepare_pretrained.py \
        --hf-repo facebook/VGGT-1B \
        --out-dir /abs/path/to/checkpoints \
        --filename vggt_1b.pt

    # Env-var overrides (handy for Docker / CI)
    HF_REPO=facebook/VGGT-1B \
    PRETRAINED_DIR=/abs/path/to/checkpoints \
    PRETRAINED_FILENAME=vggt_1b.pt \
    python data_download/prepare_pretrained.py
"""

import argparse
import os
import os.path as osp
import torch

from iggt.models.vggt import VGGT


# ── Defaults ─────────────────────────────────────────────────────────────────
# Edit these for a permanent change, or use --out-dir / PRETRAINED_DIR per run.
DEFAULT_HF_REPO = "facebook/VGGT-1B"
DEFAULT_PRETRAINED_DIR = osp.abspath(
    osp.join(osp.dirname(osp.abspath(__file__)), "..", "checkpoints")
)
DEFAULT_FILENAME = "vggt_1b.pt"
# ─────────────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--hf-repo",
        default=os.environ.get("HF_REPO", DEFAULT_HF_REPO),
        help=f"Hugging Face model id to pull (default: {DEFAULT_HF_REPO}).",
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
    return p.parse_args()


def main():
    args = parse_args()
    out_path = osp.abspath(osp.join(args.out_dir, args.filename))
    os.makedirs(osp.dirname(out_path), exist_ok=True)

    print(f"Loading {args.hf_repo} via VGGT.from_pretrained ...")
    model = VGGT.from_pretrained(args.hf_repo)
    model.eval()

    # Wrap in a dict with a "model" key so the trainer's
    # _load_resuming_checkpoint picks it up correctly. The other keys
    # (optimizer, epoch, scaler) are intentionally omitted so resume from
    # this file behaves like "load weights only, fresh optimizer state".
    state = {"model": model.state_dict()}

    n_params = sum(v.numel() for v in state["model"].values()) / 1e6
    print(f"Saving state_dict ({n_params:.1f}M params) -> {out_path}")
    torch.save(state, out_path)
    print("Done.")
    print(f"\nNext: set checkpoint.resume_checkpoint_path: {out_path} in your config.")


if __name__ == "__main__":
    main()
