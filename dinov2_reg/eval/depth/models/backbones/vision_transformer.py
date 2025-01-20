# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

#from mmcv.runner import BaseModule

from ..builder import BACKBONES
from torch import nn

@BACKBONES.register_module()
class DinoVisionTransformer(nn.module):
    """Vision Transformer."""

    def __init__(self, *args, **kwargs):
        super().__init__()