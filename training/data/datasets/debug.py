# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import os.path as osp
import logging
import numpy as np
import cv2
from PIL import Image
import random

from data.dataset_util import *
from data.base_dataset import BaseDataset


class DebugDataset(BaseDataset):
    """
    Debug dataset for testing on toy data.
    
    Reads from the toy data structure in data_debug/scene_000_extracted/frames:
    - Image/camera_0/Image_*.png
    - Depth/camera_0/Depth_*.npy
    - camview/camera_0/camview_*.npz (contains K, T, HW)
    """
    
    def __init__(
        self,
        common_conf,
        split: str = "train",
        DEBUG_DATA_DIR: str = None,
        camera_id: int = 0,
        len_train: int = 10,
        len_test: int = 5,
    ):
        """
        Initialize the DebugDataset.

        Args:
            common_conf: Configuration object with common settings.
            split (str): Dataset split, either 'train' or 'test'.
            DEBUG_DATA_DIR (str): Path to the debug data root directory (e.g., data_debug/scene_000_extracted).
            camera_id (int): Camera ID to use (default: 0).
            len_train (int): Length of the training dataset.
            len_test (int): Length of the test dataset.
        Raises:
            ValueError: If DEBUG_DATA_DIR is not specified.
        """
        super().__init__(common_conf=common_conf)

        self.debug = common_conf.debug
        self.training = common_conf.training
        self.get_nearby = common_conf.get_nearby
        self.load_depth = common_conf.load_depth
        self.inside_random = common_conf.inside_random
        self.allow_duplicate_img = common_conf.allow_duplicate_img

        if DEBUG_DATA_DIR is None:
            raise ValueError("DEBUG_DATA_DIR must be specified.")

        if split == "train":
            self.len_train = len_train
        elif split == "test":
            self.len_train = len_test
        else:
            raise ValueError(f"Invalid split: {split}")

        self.camera_id = camera_id
        self.DEBUG_DATA_DIR = DEBUG_DATA_DIR
        self.frames_dir = osp.join(DEBUG_DATA_DIR, "frames")

        # Discover all available images
        image_dir = osp.join(self.frames_dir, "Image", f"camera_{camera_id}")
        depth_dir = osp.join(self.frames_dir, "Depth", f"camera_{camera_id}")
        camview_dir = osp.join(self.frames_dir, "camview", f"camera_{camera_id}")

        if not osp.exists(image_dir):
            raise ValueError(f"Image directory not found: {image_dir}")

        # List all image files and extract frame indices
        image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
        
        if not image_files:
            raise ValueError(f"No image files found in {image_dir}")

        self.image_files = image_files
        self.image_dir = image_dir
        self.depth_dir = depth_dir
        self.camview_dir = camview_dir

        logging.info(f"Debug Dataset: Found {len(image_files)} images in {image_dir}")
        logging.info(f"Debug Dataset: Training length = {self.len_train}")

    def get_data(
        self,
        seq_index: int = None,
        img_per_seq: int = None,
        seq_name: str = None,
        ids: list = None,
        aspect_ratio: float = 1.0,
    ) -> dict:
        """
        Retrieve data for a specific sequence.

        Args:
            seq_index (int): Index of the sequence to retrieve (not used for debug dataset).
            img_per_seq (int): Number of images per sequence.
            seq_name (str): Name of the sequence (not used for debug dataset).
            ids (list): Specific frame IDs to retrieve. If None, random frames are selected.
            aspect_ratio (float): Aspect ratio for image processing.

        Returns:
            dict: A batch of data including images, depths, and other metadata.
        """
        if ids is None:
            # Randomly select frame IDs
            num_frames = len(self.image_files)
            ids = np.random.choice(
                num_frames, img_per_seq, replace=self.allow_duplicate_img
            )

        target_image_shape = self.get_target_shape(aspect_ratio)

        images = []
        depths = []
        cam_points = []
        world_points = []
        point_masks = []
        extrinsics = []
        intrinsics = []
        image_paths = []
        original_sizes = []

        for frame_id in ids:
            if frame_id >= len(self.image_files):
                continue

            image_filename = self.image_files[frame_id]
            
            # Load image
            image_path = osp.join(self.image_dir, image_filename)
            image = read_image_cv2(image_path)

            # Extract frame index from filename (format: Image_<idx>_*.png)
            parts = image_filename.split('_')
            frame_idx = int(parts[1])

            # Load depth map
            if self.load_depth:
                depth_filename = f"Depth_{frame_idx}_0_0001_0.npy"
                depth_path = osp.join(self.depth_dir, depth_filename)
                
                if osp.exists(depth_path):
                    depth_map = np.load(depth_path).astype(np.float32)
                else:
                    # Fallback: use a dummy depth map
                    logging.warning(f"Depth file not found: {depth_path}, using dummy depth")
                    depth_map = np.ones((image.shape[0], image.shape[1]), dtype=np.float32) * 2.0
            else:
                depth_map = None

            # Load camera metadata
            camview_filename = f"camview_{frame_idx}_0_0001_0.npz"
            camview_path = osp.join(self.camview_dir, camview_filename)

            if osp.exists(camview_path):
                cam_data = np.load(camview_path)
                K = cam_data['K'].astype(np.float32)  # 3x3 intrinsic matrix
                T = cam_data['T'].astype(np.float32)  # 4x4 extrinsic matrix (world-to-camera)
            else:
                raise FileNotFoundError(f"Camera metadata not found: {camview_path}")

            # Convert extrinsic from world-to-camera to camera-to-world (OpenCV convention)
            # T is world-to-camera (camera frame = T * world frame)
            # We need camera-to-world: world frame = T_inv * camera frame
            T_inv = np.linalg.inv(T)
            extri_opencv = T_inv[:3, :]  # 3x4 camera-to-world extrinsic
            intri_opencv = K  # 3x3 intrinsic

            original_size = np.array(image.shape[:2])

            # Process image and depth
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

        set_name = "debug"
        seq_name = f"camera_{self.camera_id}"

        batch = {
            "seq_name": set_name + "_" + seq_name,
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
        return batch
