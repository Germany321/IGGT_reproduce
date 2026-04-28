# Quick Reference: IGGT Debug Training

## What Was Done

Implemented a complete training pipeline for IGGT using toy data in `data_debug/`:

1. **DebugDataset** (`training/data/datasets/debug.py`) - Loads toy data
2. **IGGT Config** (`training/config/debug.yaml`) - Training configuration  
3. **Model Fix** (`iggt/models/vggt.py`) - Added `pose_enc_list` output for loss compatibility
4. **Test Script** (`test_debug_setup.py`) - Validates the setup
5. **Documentation** (`DEBUG_TRAINING_README.md`, `IMPLEMENTATION_SUMMARY.md`)

## How to Use

### Step 1: Install Package
```bash
cd /home/lhxk/workspace/repos/streamIGGT/IGGT_reproduce
pip install -e .
```

### Step 2: Test the Setup (Optional)
```bash
python test_debug_setup.py
```
This verifies that dataset, model, and loss all work together.

### Step 3: Run Training
```bash
# Single GPU
python training/launch.py --config=debug

# Multi-GPU (e.g., 2 GPUs)
torchrun --nproc_per_node=2 training/launch.py --config=debug
```

## Key Files

| File | Purpose |
|------|---------|
| `training/data/datasets/debug.py` | New debug dataset loader |
| `training/config/debug.yaml` | Debug training config (2 epochs, small batches) |
| `iggt/models/vggt.py` | Modified: added `pose_enc_list` output key |
| `test_debug_setup.py` | Validation test script |
| `DEBUG_TRAINING_README.md` | Full documentation |
| `IMPLEMENTATION_SUMMARY.md` | Implementation details |

## Data Source
- **Location**: `data_debug/scene_000_extracted/frames/`
- **Contains**: 
  - 101 RGB images (512×288 px)
  - Depth maps (float32)
  - Camera intrinsics and extrinsics

## Configuration Highlights

```yaml
model: iggt.models.vggt.IGGT         # IGGT model with instance head
dataset: DebugDataset                # Loads toy data
epochs: 2                            # Short for quick testing
batches/epoch: 10                    # Small for quick iteration
batch_size: 4                        # Small GPU batch
loss: VGGT loss (camera + depth)    # No MVC loss yet
```

## What It Does

### Training Loop
1. Loads batch of 4 images from toy dataset
2. Runs IGGT model forward pass → predicts:
   - Camera poses (pose_enc)
   - Depth maps (depth)
   - 3D world points (world_points)
   - Instance features (part_feat)
3. Computes losses:
   - Camera pose L1 loss
   - Depth regression + gradient + confidence loss
4. Backprop and update parameters

### Expected Runtime
- Per epoch: ~10 seconds (10 batches × ~1 sec)
- Full training: ~20-30 seconds (2 epochs)
- Checkpoints saved to: `logs/debug_exp/ckpts/`

## Architecture

```
Image Input [B=1, S=4, H=288, W=512]
     ↓
Aggregator (unified transformer)
     ↓
Heads:
  ├─ CameraHead  → pose_enc [B, S, 9]
  ├─ DepthHead   → depth [B, S, H, W, 1]
  ├─ PointHead   → world_points [B, S, H, W, 3]
  └─ PartHead    → part_feat [B, S, 8, H, W] (instance)
     ↓
Loss Function
     ↓
Backprop & Update
```

## Next Steps After Validation

1. **Add MVC Loss** - For instance discrimination
   - Extend dataset with instance IDs
   - Implement contrastive loss term
   
2. **Use Full Datasets** - Co3D, VKitti, etc.
   - Update config to point to real data
   
3. **Fine-tune Hyperparameters** - Learning rate, weights, etc.

## Common Issues

| Issue | Solution |
|-------|----------|
| Module not found | Run `pip install -e .` first |
| Dataset not found | Check `data_debug/scene_000_extracted/frames/` exists |
| CUDA out of memory | Reduce `max_img_per_gpu` in config |
| NaN losses | Check depth value ranges in dataset |

## Code Changes Summary

✓ Non-destructive (VGGT code not modified)  
✓ Only added 2 lines to IGGT forward (pose_enc_list key)  
✓ All new code in separate files  
✓ Backward compatible with existing training

## Support

- Full documentation: `DEBUG_TRAINING_README.md`
- Implementation details: `IMPLEMENTATION_SUMMARY.md`
- Test script: `test_debug_setup.py`
