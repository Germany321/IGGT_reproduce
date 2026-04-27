# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import logging
from typing import (
    Any,
    Dict,
    List,
)

import torch
import torch.nn as nn
import os
from iopath.common.file_io import g_pathmgr




class DDPCheckpointSaver:
    def __init__(
        self,
        checkpoint_folder: str,
        checkpoint_names: List[str],
        rank: int,
        epoch: int,
    ):
        super().__init__()
        self.checkpoint_folder = checkpoint_folder
        self.checkpoint_names = checkpoint_names
        self.worker_id = rank
        self.epoch = epoch

    def save_checkpoint(
        self,
        model: nn.Module,
        **kwargs: Any,
    ) -> None:
        if self.worker_id != 0:
            return

        # Pull weights to CPU before pickling — torch.save on CUDA tensors
        # transiently doubles the GPU footprint of the state dict.
        checkpoint = dict(**kwargs)
        checkpoint["model"] = {
            k: v.detach().cpu() if isinstance(v, torch.Tensor) and v.is_cuda else v
            for k, v in model.state_dict().items()
        }

        for ckpt_name in self.checkpoint_names:
            checkpoint_path = os.path.join(
                self.checkpoint_folder, f"{ckpt_name}.pt"
            )
            logging.info(
                f"Saving checkpoint at epoch {self.epoch} to {checkpoint_path}"
            )
            robust_torch_save(checkpoint, checkpoint_path)



def robust_torch_save(checkpoint: Dict[str, Any], checkpoint_path: str) -> None:
    """
    A more robust version of torch.save that works better with preemptions
    and corruptions if a job is preempted during save.
    """
    # Move the existing checkpoint to a backup location
    backup_checkpoint_path = checkpoint_path + ".bak"
    backup_checkpoint_path_saved = False
    if g_pathmgr.exists(checkpoint_path):
        assert not g_pathmgr.exists(
            backup_checkpoint_path
        ), f"this should not exist... {backup_checkpoint_path}"
        g_pathmgr.mv(checkpoint_path, backup_checkpoint_path)
        backup_checkpoint_path_saved = True
    # Save the checkpoint
    with g_pathmgr.open(checkpoint_path, "wb") as f:
        torch.save(checkpoint, f)
    # Remove the backup checkpoint
    if backup_checkpoint_path_saved:
        g_pathmgr.rm(backup_checkpoint_path)