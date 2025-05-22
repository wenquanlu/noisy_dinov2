# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn

#from ...models.builder import LOSSES


#@LOSSES.register_module()
class MSELoss(nn.Module):
    """SigLoss.

        This follows `AdaBins <https://arxiv.org/abs/2011.14141>`_.

    Args:
        valid_mask (bool): Whether filter invalid gt (gt > 0). Default: True.
        loss_weight (float): Weight of the loss. Default: 1.0.
        max_depth (int): When filtering invalid gt, set a max threshold. Default: None.
        warm_up (bool): A simple warm up stage to help convergence. Default: False.
        warm_iter (int): The number of warm up stage. Default: 100.
    """

    def __init__(
        self):
        super(MSELoss, self).__init__()
        self.loss_name = "mse_loss"
        self.loss = nn.MSELoss()


    def forward(self, depth_pred, depth_gt):
        """Forward function."""
        return self.loss(depth_pred, depth_gt)
