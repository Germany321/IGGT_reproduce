# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import os.path as osp
import logging
import hashlib
import random

import cv2
import numpy as np

from data.dataset_util import *
from data.base_dataset import BaseDataset


class ScanNetPPDataset(BaseDataset):
    """
    ScanNet++ iPhone scenes (processed_scannetpp_v2 layout).

    Expected layout (output of data_download/scannetpp/unzip.py):

        SCANNETPP_DIR/
            <scene_id>/
                depth/frame_NNNNNN.png            # uint16, depth_mm = pixel value
                images/frame_NNNNNN.jpg           # RGB, 690x920
                refined_ins_ids/
                    frame_NNNNNN.jpg.npy          # int16 (H, W) instance ids matching image
                    frame_NNNNNN.png              # high-res variant — ignored
                scene_iphone_metadata.npz         # trajectories (N,4,4) cam->world,
                                                  # intrinsics (N,3,3),
                                                  # images (N,) filename strings
            <scene_id>/...

    One scene = one sequence. Trajectories are stored camera-to-world (standard
    ScanNet++ convention) and used directly as the OpenCV extrinsic; depths are
    uint16 millimetres (divide by ``depth_scale`` = 1000 to get metres).
    """

    def __init__(
        self,
        common_conf,
        split: str = "train",
        SCANNETPP_DIR: str = None,
        len_train: int = 100000,
        len_test: int = 10000,
        min_num_images: int = 8,
        test_fraction: float = 0.05,
        depth_scale: float = 1000.0,
        load_instance_seg: bool = True,
        # Subsample per-scene frames to avoid 600+ frames per scene blowing
        # up the index. Pick every Nth frame; use 1 to keep all.
        frame_stride: int = 1,
    ):
        super().__init__(common_conf=common_conf)

        self.debug = common_conf.debug
        self.training = common_conf.training
        self.get_nearby = common_conf.get_nearby
        self.load_depth = common_conf.load_depth
        self.inside_random = common_conf.inside_random
        self.allow_duplicate_img = common_conf.allow_duplicate_img

        if SCANNETPP_DIR is None:
            raise ValueError("SCANNETPP_DIR must be specified.")
        if not osp.isdir(SCANNETPP_DIR):
            raise FileNotFoundError(f"SCANNETPP_DIR does not exist: {SCANNETPP_DIR}")

        if split == "train":
            self.len_train = len_train
        elif split == "test":
            self.len_train = len_test
        else:
            raise ValueError(f"Invalid split: {split}")

        self.SCANNETPP_DIR = SCANNETPP_DIR
        self.depth_scale = depth_scale
        self.load_instance_seg = load_instance_seg
        self.frame_stride = max(1, int(frame_stride))

        # ── Discover scenes ──────────────────────────────────────────────
        # data_store[scene_id] = {
        #   "scene_dir": str,
        #   "frame_names": [str, ...],   # without extension, e.g. "frame_000000"
        #   "trajectories": np.ndarray,  # (N, 4, 4) cam->world
        #   "intrinsics":   np.ndarray,  # (N, 3, 3)
        # }
        self.data_store = {}
        scene_ids = sorted(
            d for d in os.listdir(SCANNETPP_DIR)
            if osp.isdir(osp.join(SCANNETPP_DIR, d))
        )

        total_frame_num = 0
        for scene_id in scene_ids:
            scene_dir = osp.join(SCANNETPP_DIR, scene_id)
            meta_path = osp.join(scene_dir, "scene_iphone_metadata.npz")
            image_dir = osp.join(scene_dir, "images")
            if not (osp.exists(meta_path) and osp.isdir(image_dir)):
                continue

            # Train/test split — by hash of scene_id, identical across ranks.
            h = int(hashlib.md5(scene_id.encode()).hexdigest(), 16)
            in_test = (h % 10000) / 10000.0 < test_fraction
            if (split == "test") != in_test:
                continue

            try:
                meta = np.load(meta_path, allow_pickle=True)
                images = meta["images"]                # (N,) str
                trajectories = meta["trajectories"]    # (N, 4, 4)
                intrinsics = meta["intrinsics"]        # (N, 3, 3)
            except Exception as e:
                logging.warning(f"Skipping {scene_id}: cannot read metadata ({e})")
                continue

            # frame_names align 1:1 with trajectories/intrinsics row index.
            frame_names = [osp.splitext(str(s))[0] for s in images]

            if self.frame_stride > 1:
                idx = np.arange(0, len(frame_names), self.frame_stride)
                frame_names = [frame_names[i] for i in idx]
                trajectories = trajectories[idx]
                intrinsics = intrinsics[idx]

            if len(frame_names) < min_num_images:
                continue

            self.data_store[scene_id] = {
                "scene_dir": scene_dir,
                "frame_names": frame_names,
                "trajectories": trajectories.astype(np.float32),
                "intrinsics": intrinsics.astype(np.float32),
            }
            total_frame_num += len(frame_names)

        self.sequence_list = list(self.data_store.keys())
        self.sequence_list_len = len(self.sequence_list)
        self.total_frame_num = total_frame_num

        if self.sequence_list_len == 0:
            raise RuntimeError(
                f"No ScanNet++ scenes found under {SCANNETPP_DIR} for split={split}, "
                f"min_num_images={min_num_images}."
            )

        status = "Training" if self.training else "Testing"
        logging.info(
            f"{status}: ScanNet++ Data size: {self.sequence_list_len} scenes "
            f"({self.total_frame_num} frames) | split={split} stride={self.frame_stride}"
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
        scene_dir = meta["scene_dir"]
        frame_names = meta["frame_names"]
        trajectories = meta["trajectories"]
        intrinsics = meta["intrinsics"]

        if ids is None:
            ids = np.random.choice(
                len(frame_names), img_per_seq, replace=self.allow_duplicate_img
            )

        target_image_shape = self.get_target_shape(aspect_ratio)

        image_dir = osp.join(scene_dir, "images")
        depth_dir = osp.join(scene_dir, "depth")
        seg_dir = osp.join(scene_dir, "refined_ins_ids")

        images, depths = [], []
        cam_points, world_points, point_masks = [], [], []
        extrinsics, intri_list = [], []
        image_paths, original_sizes = [], []
        instance_segs = []

        for sel in ids:
            sel = int(sel)
            name = frame_names[sel]

            image_path = osp.join(image_dir, f"{name}.jpg")
            image = read_image_cv2(image_path)

            if self.load_depth:
                depth_path = osp.join(depth_dir, f"{name}.png")
                if osp.exists(depth_path):
                    raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
                    if raw is None:
                        logging.warning(f"Depth read failed: {depth_path}")
                        depth_map = np.ones(image.shape[:2], dtype=np.float32) * 2.0
                    else:
                        depth_map = raw.astype(np.float32) / float(self.depth_scale)
                        # Resize depth to image resolution if metadata mismatch.
                        if depth_map.shape != image.shape[:2]:
                            depth_map = cv2.resize(
                                depth_map, (image.shape[1], image.shape[0]),
                                interpolation=cv2.INTER_NEAREST,
                            )
                else:
                    logging.warning(f"Depth file not found: {depth_path}, dummy=2m")
                    depth_map = np.ones(image.shape[:2], dtype=np.float32) * 2.0
            else:
                depth_map = None

            # Trajectory is camera-to-world; matches OpenCV extrinsic convention
            # used elsewhere in this codebase (3x4 cam2world).
            extri_opencv = trajectories[sel][:3, :].astype(np.float32)
            intri_opencv = intrinsics[sel].astype(np.float32)

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
            intri_list.append(intri_opencv)
            cam_points.append(cam_coords_points)
            world_points.append(world_coords_points)
            point_masks.append(point_mask)
            image_paths.append(image_path)
            original_sizes.append(original_size)

            if self.load_instance_seg:
                seg = self._load_instance_seg(seg_dir, name, image.shape[:2])
                instance_segs.append(seg)

        batch = {
            "seq_name": "scannetpp_" + seq_name,
            "ids": ids,
            "frame_num": len(extrinsics),
            "images": images,
            "depths": depths,
            "extrinsics": extrinsics,
            "intrinsics": intri_list,
            "cam_points": cam_points,
            "world_points": world_points,
            "point_masks": point_masks,
            "original_sizes": original_sizes,
        }
        if self.load_instance_seg and len(instance_segs) > 0:
            batch["instance_seg"] = instance_segs
        return batch

    # ──────────────────────────────────────────────────────────────────────
    def _load_instance_seg(self, seg_dir, frame_name, target_hw):
        """ScanNet++ instance ids are stored in `refined_ins_ids/<name>.jpg.npy`
        already at image resolution (H, W) int16. Resize with nearest-neighbor
        to the post-processing image shape.
        """
        seg_path = osp.join(seg_dir, f"{frame_name}.jpg.npy")
        if not osp.exists(seg_path):
            logging.warning(f"Seg file not found: {seg_path}, returning zeros")
            return np.zeros(target_hw, dtype=np.int32)

        seg = np.load(seg_path).astype(np.int32)
        H_t, W_t = target_hw
        if seg.shape != (H_t, W_t):
            seg = cv2.resize(seg, (W_t, H_t), interpolation=cv2.INTER_NEAREST)
        return seg.astype(np.int32)
