#!/usr/bin/env python3
"""
Quick debug script to test the debug dataset, IGGT model, and loss function.
Run from the repo root: python test_debug_setup.py
"""

import sys
import os
import logging
import torch
from hydra import initialize, compose
from hydra.utils import instantiate
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_debug_setup():
    """Test dataset loading, model forward pass, and loss computation."""
    
    logger.info("=" * 60)
    logger.info("Testing Debug Setup: Dataset + IGGT Model + Loss")
    logger.info("=" * 60)
    
    # Initialize Hydra with the debug config
    with initialize(version_base=None, config_path="training/config"):
        cfg = compose(config_name="debug")
        
        logger.info(f"\nConfig: {cfg.exp_name}")
        logger.info(f"Model: {cfg.model._target_}")
        
        # 1. Test dataset loading
        logger.info("\n[1/3] Testing dataset loading...")
        try:
            # Get dataset config
            dataset_cfg = cfg.data.train.dataset.dataset_configs[0]
            logger.info(f"  Dataset: {dataset_cfg._target_}")
            logger.info(f"  Debug data dir: {dataset_cfg.DEBUG_DATA_DIR}")
            
            # Instantiate dataset
            dataset = instantiate(dataset_cfg)
            logger.info(f"  ✓ Dataset instantiated successfully")
            logger.info(f"    Dataset length: {len(dataset)}")
            
            # Get a sample
            sample_idx = (0, 4, 1.0)  # (seq_index, img_per_seq, aspect_ratio)
            batch = dataset[sample_idx]
            logger.info(f"  ✓ Got sample batch")
            logger.info(f"    Batch keys: {list(batch.keys())}")
            logger.info(f"    Frame count: {batch['frame_num']}")
            logger.info(f"    Image shapes: {[img.shape for img in batch['images']]}")
            
        except Exception as e:
            logger.error(f"  ✗ Dataset loading failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # 2. Test model forward pass
        logger.info("\n[2/3] Testing model forward pass...")
        try:
            # Instantiate model
            model = instantiate(cfg.model)
            model.eval()
            logger.info(f"  ✓ Model instantiated successfully")
            
            # Prepare input images
            images_list = batch['images']
            images_tensor = torch.stack([torch.from_numpy(img).float() for img in images_list])
            images_tensor = images_tensor.unsqueeze(0)  # Add batch dimension
            logger.info(f"  Input shape: {images_tensor.shape}")
            
            # Forward pass
            with torch.no_grad():
                predictions = model(images=images_tensor)
            
            logger.info(f"  ✓ Forward pass successful")
            logger.info(f"    Prediction keys: {list(predictions.keys())}")
            
            # Check critical keys
            critical_keys = ["pose_enc", "pose_enc_list", "depth", "world_points", "images"]
            for key in critical_keys:
                if key in predictions:
                    val = predictions[key]
                    if isinstance(val, torch.Tensor):
                        logger.info(f"    {key}: shape={tuple(val.shape)}")
                    elif isinstance(val, list):
                        logger.info(f"    {key}: list of {len(val)} items")
                else:
                    logger.warning(f"    {key}: MISSING")
            
        except Exception as e:
            logger.error(f"  ✗ Model forward pass failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # 3. Test loss computation
        logger.info("\n[3/3] Testing loss computation...")
        try:
            # Instantiate loss function
            loss_fn = instantiate(cfg.loss)
            logger.info(f"  ✓ Loss function instantiated")
            
            # Prepare batch for loss (convert numpy arrays to tensors)
            batch_tensor = {}
            for key, value in batch.items():
                if isinstance(value, list):
                    if len(value) > 0 and isinstance(value[0], np.ndarray):
                        batch_tensor[key] = [torch.from_numpy(v).float() for v in value]
                    else:
                        batch_tensor[key] = value
                elif isinstance(value, np.ndarray):
                    batch_tensor[key] = torch.from_numpy(value).float()
                else:
                    batch_tensor[key] = value
            
            # Compute loss
            loss_dict = loss_fn(predictions, batch_tensor)
            logger.info(f"  ✓ Loss computation successful")
            logger.info(f"    Loss keys: {list(loss_dict.keys())}")
            
            # Log loss values
            for key, val in loss_dict.items():
                if isinstance(val, torch.Tensor):
                    logger.info(f"    {key}: {val.item():.6f}")
                else:
                    logger.info(f"    {key}: {val}")
            
        except Exception as e:
            logger.error(f"  ✗ Loss computation failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        logger.info("\n" + "=" * 60)
        logger.info("✓ All tests passed!")
        logger.info("=" * 60)
        return True


if __name__ == "__main__":
    success = test_debug_setup()
    sys.exit(0 if success else 1)
