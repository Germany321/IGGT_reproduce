# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import os.path as osp
import re
import json
import random
import logging
import hashlib

import cv2
import numpy as np

from data.dataset_util import *
from data.base_dataset import BaseDataset


# Filenames produced by Infinigen extraction follow the pattern
#   <prefix>_<frame>_0_0001_0.<ext>
# e.g. Image_42_0_0001_0.png, Depth_42_0_0001_0.npy, camview_42_0_0001_0.npz.
_FRAME_RE = re.compile(r".*?_(\d+)_0_0001_0\.[A-Za-z0-9]+$")


def _frame_idx_from_name(name: str):
    m = _FRAME_RE.match(name)
    return int(m.group(1)) if m else None


class InfinigenDataset(BaseDataset):
    """
    Infinigen multi-view dataset.

    Expected layout (output of data_download/infinigen/unzip.py):

        INFINIGEN_DIR/
            scene_000/
                <sub_id_a>/frames/{Image,Depth,camview,Objects,ObjectSegmentation}/camera_K/...
                <sub_id_b>/frames/...
            scene_001/
                ...

    Each (scene_id, sub_id, camera_id) tuple is treated as one sequence —
    a single camera trajectory whose frames share intrinsics layout and an
    instance-segmentation id space.
    """

    def __init__(
        self,
        common_conf,
        split: str = "train",
        INFINIGEN_DIR: str = None,
        camera_id: int = 0,
        len_train: int = 100000,
        len_test: int = 10000,
        min_num_images: int = 8,
        test_fraction: float = 0.05,
        load_instance_seg: bool = True,
        mesh_only: bool = True,
    ):
        super().__init__(common_conf=common_conf)

        self.debug = common_conf.debug
        self.training = common_conf.training
        self.get_nearby = common_conf.get_nearby
        self.load_depth = common_conf.load_depth
        self.inside_random = common_conf.inside_random
        self.allow_duplicate_img = common_conf.allow_duplicate_img

        if INFINIGEN_DIR is None:
            raise ValueError("INFINIGEN_DIR must be specified.")
        if not osp.isdir(INFINIGEN_DIR):
            raise FileNotFoundError(f"INFINIGEN_DIR does not exist: {INFINIGEN_DIR}")

        if split == "train":
            self.len_train = len_train
        elif split == "test":
            self.len_train = len_test
        else:
            raise ValueError(f"Invalid split: {split}")

        self.INFINIGEN_DIR = INFINIGEN_DIR
        self.camera_id = camera_id
        self.min_num_images = min_num_images
        self.load_instance_seg = load_instance_seg
        self.mesh_only = mesh_only

        # ── Discover sequences ────────────────────────────────────────────
        # data_store[seq_name] = {
        #   "scene": str, "sub_id": str, "camera_id": int,
        #   "frames_dir": str, "frame_ids": [int, ...],
        # }
        self.data_store = {}
        scene_dirs = sorted(
            d for d in os.listdir(INFINIGEN_DIR)
            if d.startswith("scene_") and osp.isdir(osp.join(INFINIGEN_DIR, d))
        )

        total_frame_num = 0
        for scene in scene_dirs:
            scene_path = osp.join(INFINIGEN_DIR, scene)
            for sub_id in sorted(os.listdir(scene_path)):
                sub_path = osp.join(scene_path, sub_id)
                frames_dir = osp.join(sub_path, "frames")
                image_dir = osp.join(frames_dir, "Image", f"camera_{camera_id}")
                if not osp.isdir(image_dir):
                    continue

                frame_ids = []
                for f in os.listdir(image_dir):
                    if not f.endswith(".png"):
                        continue
                    fid = _frame_idx_from_name(f)
                    if fid is not None:
                        frame_ids.append(fid)
                if len(frame_ids) < min_num_images:
                    continue
                frame_ids.sort()

                seq_name = f"{scene}/{sub_id}/cam{camera_id}"
                # Deterministic hash-based train/test split so the same
                # sequence always lands in the same split across ranks/runs.
                h = int(hashlib.md5(seq_name.encode()).hexdigest(), 16)
                bucket = (h % 10000) / 10000.0  # [0, 1)
                in_test = bucket < test_fraction
                if (split == "test") != in_test:
                    continue

                self.data_store[seq_name] = {
                    "scene": scene,
                    "sub_id": sub_id,
                    "camera_id": camera_id,
                    "frames_dir": frames_dir,
                    "frame_ids": frame_ids,
                }
                total_frame_num += len(frame_ids)

        self.sequence_list = list(self.data_store.keys())
        self.sequence_list_len = len(self.sequence_list)
        self.total_frame_num = total_frame_num

        if self.sequence_list_len == 0:
            raise RuntimeError(
                f"No Infinigen sequences found under {INFINIGEN_DIR} for split={split}, "
                f"camera_id={camera_id}, min_num_images={min_num_images}."
            )

        status = "Training" if self.training else "Testing"
        logging.info(
            f"{status}: Infinigen Data size: {self.sequence_list_len} sequences "
            f"({self.total_frame_num} frames) | split={split} cam={camera_id}"
        )

    # ──────────────────────────────────────────────────────────────────────
    def get_data(
        self,
        seq_index: int = None,
        img_per_seq: int = None,
        seq_name: str = None,
        ids: list = None,
        aspect_ratio: float = 1.0,
    ) -> dict:
        if self.inside_random:
            seq_index = random.randint(0, self.sequence_list_len - 1)

        if seq_name is None:
            seq_name = self.sequence_list[seq_index % self.sequence_list_len]

        meta = self.data_store[seq_name]
        frame_ids = meta["frame_ids"]
        frames_dir = meta["frames_dir"]
        camera_id = meta["camera_id"]

        if ids is None:
            ids = np.random.choice(
                len(frame_ids), img_per_seq, replace=self.allow_duplicate_img
            )

        target_image_shape = self.get_target_shape(aspect_ratio)

        image_dir = osp.join(frames_dir, "Image", f"camera_{camera_id}")
        depth_dir = osp.join(frames_dir, "Depth", f"camera_{camera_id}")
        camview_dir = osp.join(frames_dir, "camview", f"camera_{camera_id}")
        seg_dir = osp.join(frames_dir, "ObjectSegmentation", f"camera_{camera_id}")
        objects_dir = osp.join(frames_dir, "Objects", f"camera_{camera_id}")

        images, depths = [], []
        cam_points, world_points, point_masks = [], [], []
        extrinsics, intrinsics = [], []
        image_paths, original_sizes = [], []
        instance_segs = []

        for sel in ids:
            frame_idx = int(frame_ids[int(sel)])

            image_path = osp.join(image_dir, f"Image_{frame_idx}_0_0001_0.png")
            image = read_image_cv2(image_path)

            if self.load_depth:
                depth_path = osp.join(depth_dir, f"Depth_{frame_idx}_0_0001_0.npy")
                if osp.exists(depth_path):
                    depth_map = np.load(depth_path).astype(np.float32)
                else:
                    logging.warning(f"Depth file not found: {depth_path}, using dummy depth")
                    depth_map = np.ones(image.shape[:2], dtype=np.float32) * 2.0
            else:
                depth_map = None

            camview_path = osp.join(camview_dir, f"camview_{frame_idx}_0_0001_0.npz")
            if not osp.exists(camview_path):
                raise FileNotFoundError(f"Camera metadata not found: {camview_path}")
            cam_data = np.load(camview_path)
            K = cam_data["K"].astype(np.float32)
            T = cam_data["T"].astype(np.float32)  # world-to-camera

            T_inv = np.linalg.inv(T)
            extri_opencv = T_inv[:3, :]
            intri_opencv = K

            original_size = np.array(image.shape[:2])

            (
                image,
                depth_map,
                extri_opencv,
                intri_opencv,
                world_coords_points,
                cam_coords_points,
                point_mask,
                _,
            ) = self.process_one_image(
                image,
                depth_map,
                extri_opencv,
                intri_opencv,
                original_size,
                target_image_shape,
                filepath=image_path,
            )

            images.append(image)
            depths.append(depth_map)
            extrinsics.append(extri_opencv)
            intrinsics.append(intri_opencv)
            cam_points.append(cam_coords_points)
            world_points.append(world_coords_points)
            point_masks.append(point_mask)
            image_paths.append(image_path)
            original_sizes.append(original_size)

            if self.load_instance_seg:
                seg = self._load_instance_seg(
                    seg_dir, objects_dir, frame_idx, image.shape[:2]
                )
                instance_segs.append(seg)

        batch = {
            "seq_name": "infinigen_" + seq_name,
            "ids": ids,
            "frame_num": len(extrinsics),
            "images": images,
            "depths": depths,
            "extrinsics": extrinsics,
            "intrinsics": intrinsics,
            "cam_points": cam_points,
            "world_points": world_points,
            "point_masks": point_masks,
            "original_sizes": original_sizes,
        }
        if self.load_instance_seg and len(instance_segs) > 0:
            batch["instance_seg"] = instance_segs
        return batch

    # ──────────────────────────────────────────────────────────────────────
    def _load_instance_seg(self, seg_dir, objects_dir, frame_idx, target_hw):
        seg_path = osp.join(seg_dir, f"ObjectSegmentation_{frame_idx}_0_0001_0.npy")
        if not osp.exists(seg_path):
            logging.warning(f"Seg file not found: {seg_path}, returning zeros")
            return np.zeros(target_hw, dtype=np.int32)

        seg = np.load(seg_path).astype(np.int32)

        if self.mesh_only:
            keep_ids = self._mesh_object_indices(objects_dir, frame_idx)
            if keep_ids is not None:
                keep_mask = np.isin(seg, list(keep_ids))
                seg = np.where(keep_mask, seg, 0)

        H_t, W_t = target_hw
        if seg.shape != (H_t, W_t):
            seg = cv2.resize(seg, (W_t, H_t), interpolation=cv2.INTER_NEAREST)
        return seg.astype(np.int32)

    def _mesh_object_indices(self, objects_dir, frame_idx):
        obj_path = osp.join(objects_dir, f"Objects_{frame_idx}_0_0001_0.json")
        if not osp.exists(obj_path):
            return None
        try:
            with open(obj_path) as f:
                meta = json.load(f)
        except Exception:
            return None
        return {v["object_index"] for v in meta.values()
                if isinstance(v, dict) and v.get("type") == "MESH"
                and "object_index" in v}
