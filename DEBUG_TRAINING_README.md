# Debug Training Setup for IGGT

This document describes the debug training setup that was implemented for the IGGT model using the toy data in `data_debug/scene_000_extracted`.

## What Was Implemented

### 1. New Debug Dataset Class (`training/data/datasets/debug.py`)

A new `DebugDataset` class was created to load data from the toy scene in `data_debug/scene_000_extracted/frames`. This dataset:

- **Loads images** from `frames/Image/camera_0/*.png`
- **Loads depth maps** from `frames/Depth/camera_0/*.npy`
- **Loads camera metadata** from `frames/camview/camera_0/*.npz` (contains intrinsics K, extrinsics T, and image size HW)
- **Inherits from BaseDataset** to reuse image processing pipeline (resizing, augmentation, coordinate transformation)
- **Returns batches** with the same structure as Co3dDataset for compatibility with the training pipeline

Key features:
- Converts world-to-camera extrinsics to camera-to-world convention (OpenCV standard)
- Properly handles frame indexing and metadata file lookup
- Supports random frame selection for variable-length sequences

### 2. IGGT Model Class Enhancements

The existing `IGGT` model class in `iggt/models/vggt.py` was enhanced to:

- **Output `pose_enc_list` key** in addition to existing `pose_enc` for loss function compatibility
- This ensures the loss function in `training/loss.py` can properly detect and compute camera pose loss

Changes made:
- Added `predictions["pose_enc_list"] = pose_enc_list` output alongside existing keys
- Applied same fix to VGGT class for consistency

### 3. Debug Configuration (`training/config/debug.yaml`)

A new debug configuration file was created with:

- **Model**: Uses `iggt.models.vggt.IGGT` (instance-grounded geometry transformer)
- **Dataset**: Uses `DebugDataset` pointing to `data_debug/scene_000_extracted`
- **Training parameters optimized for quick iterations**:
  - `max_epochs: 2` (short training for testing)
  - `limit_train_batches: 10` (only 10 batches per epoch)
  - `limit_val_batches: 5` (small validation set)
  - `max_img_per_gpu: 4` (small batch size)
  - `num_workers: 0` (no multiprocessing for debugging)
- **Loss function**: Uses existing VGGT loss (camera + depth), no MVC loss yet
- **Optimizer**: Standard AdamW with reduced learning rate (1e-4)

## File Changes Summary

| File | Change Type | Description |
|------|------------|-------------|
| `training/data/datasets/debug.py` | **NEW** | Debug dataset class for toy data |
| `training/config/debug.yaml` | **NEW** | Debug training configuration |
| `iggt/models/vggt.py` | **MODIFIED** | Added `pose_enc_list` output key for both VGGT and IGGT |
| `setup.py` | **NEW** (from previous session) | Package installation file |

## How to Use

### 1. Install the Package

```bash
pip install -e .
```

### 2. Run Training with Debug Config

```bash
# Single GPU training
python training/launch.py --config-name=debug

# Multi-GPU training (e.g., 2 GPUs)
torchrun --nproc_per_node=2 training/launch.py --config-name=debug
```

### 3. Test the Setup

A test script is provided to verify dataset loading, model forward pass, and loss computation:

```bash
python test_debug_setup.py
```

The script will:
1. Load the debug dataset
2. Run a forward pass through the IGGT model
3. Compute losses
4. Report success/failure and intermediate values

## Expected Behavior

- **Dataset loading**: Should load ~101 images and depth maps from `data_debug/scene_000_extracted`
- **Model forward pass**: IGGT model outputs:
  - `pose_enc` and `pose_enc_list`: Camera pose encoding [B, S, 9]
  - `depth` and `depth_conf`: Depth maps and confidence [B, S, H, W]
  - `world_points` and `world_points_conf`: 3D world coordinates
  - `part_feat`: Instance features [B, S, 8, H, W]
  - `images`: Input images
- **Loss computation**: 
  - Camera loss: Computed from pose predictions
  - Depth loss: Computed from depth predictions
  - Total objective loss: Sum of all component losses

## Data Structure

### Debug Data Layout

```
data_debug/scene_000_extracted/frames/
├── Image/camera_0/
│   └── Image_*.png          # RGB images (512×288)
├── Depth/camera_0/
│   ├── Depth_*.npy          # Depth maps (288×512) as float32
│   └── Depth_*.png          # Visualization
├── camview/camera_0/
│   └── camview_*.npz        # Camera metadata (K, T, HW)
├── ObjectSegmentation/
└── Objects/
```

### Batch Dictionary Structure

The dataset returns batches with:

```python
batch = {
    'seq_name': str,                      # Scene identifier
    'ids': np.ndarray,                    # Frame indices
    'frame_num': int,                     # Number of frames
    'images': list[np.ndarray],           # Processed images
    'depths': list[np.ndarray],           # Processed depth maps
    'extrinsics': list[np.ndarray],       # 3×4 camera-to-world matrices
    'intrinsics': list[np.ndarray],       # 3×3 intrinsic matrices
    'cam_points': list[np.ndarray],       # 3D points in camera frame
    'world_points': list[np.ndarray],     # 3D points in world frame
    'point_masks': list[np.ndarray],      # Valid depth mask
    'original_sizes': list[np.ndarray],   # Original image sizes
}
```

## Next Steps

After verifying the debug setup works:

1. **Add MVC Loss**: Implement multi-view contrastive loss term (not in current setup)
   - Add instance ID/mask fields to dataset
   - Implement contrastive loss function in `training/loss.py`
   - Add hyperparameters (λ_pull, λ_push, margin M) to config

2. **Extend to Additional Datasets**: Update `ComposedDataset` to include DebugDataset alongside Co3D

3. **Production Training**: Once validated, switch to full datasets (Co3D, VKitti, etc.)

## Key Design Decisions

- **Non-destructive implementation**: VGGT code remains untouched; IGGT uses existing architecture
- **Debug dataset as standalone class**: Allows independent testing and validation
- **Loss compatibility fix**: Added `pose_enc_list` key without removing `pose_enc` to maintain backward compatibility
- **Minimal config**: Debug config optimized for fast iteration, not accuracy

## Troubleshooting

### Issue: Dataset not found
- Ensure `data_debug/scene_000_extracted/frames/` directory exists
- Check that Image, Depth, and camview subdirectories are populated

### Issue: Model forward pass fails
- Verify IGGT model class imports correctly from `iggt.models.vggt`
- Check that aggregator, heads, and adaptor modules are properly installed

### Issue: Loss computation returns NaN
- Check that batch contains valid values (no infinite depths)
- Ensure point_masks are properly computed for valid depth thresholding
- Verify extrinsics and intrinsics are reasonable

## Files to Review

- `training/data/datasets/debug.py` - Debug dataset implementation
- `training/config/debug.yaml` - Debug training configuration
- `iggt/models/vggt.py` - VGGT/IGGT model definitions
- `training/loss.py` - Loss function implementation
- `test_debug_setup.py` - Verification test script
