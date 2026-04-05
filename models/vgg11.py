"""VGG11 encoder — implemented from scratch per arxiv:1409.1556.

Architecture (8 conv layers + 3 FC layers = 11 weight layers):
  Block 1: Conv(3→64,   3×3, p=1) + BN + ReLU → MaxPool(2,2) → 112×112×64
  Block 2: Conv(64→128, 3×3, p=1) + BN + ReLU → MaxPool(2,2) → 56×56×128
  Block 3: Conv(128→256,3×3, p=1) + BN + ReLU
           Conv(256→256,3×3, p=1) + BN + ReLU → MaxPool(2,2) → 28×28×256
  Block 4: Conv(256→512,3×3, p=1) + BN + ReLU
           Conv(512→512,3×3, p=1) + BN + ReLU → MaxPool(2,2) → 14×14×512
  Block 5: Conv(512→512,3×3, p=1) + BN + ReLU
           Conv(512→512,3×3, p=1) + BN + ReLU → MaxPool(2,2) → 7×7×512

BatchNorm placement rationale:
  BN is inserted after Conv, before ReLU. This normalises pre-activations,
  which stabilises the gradient signal and allows higher learning rates.
  With BN, bias=False in Conv2d because BN's beta parameter is equivalent.

Skip connection design:
  When return_features=True, the output of each block (after MaxPool) is
  returned as a dict. These are used as skip connections in the U-Net decoder
  (Task 3). The skip features are captured post-MaxPool so they match the
  spatial dimensions the decoder expects at each upsampling stage.
"""

from typing import Dict, Tuple, Union

import torch
import torch.nn as nn

from .layers import CustomDropout


def _conv_bn_relu(in_c: int, out_c: int) -> nn.Sequential:
    """Standard VGG conv unit: Conv(3×3) → BatchNorm → ReLU."""
    return nn.Sequential(
        # padding=1 preserves spatial size through each conv (same-padding)
        # bias=False because BatchNorm already has a learnable bias (beta)
        nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    )


class VGG11Encoder(nn.Module):
    """VGG11-style convolutional backbone with optional skip-connection returns."""

    def __init__(self, in_channels: int = 3):
        """
        Args:
            in_channels: Number of input image channels (3 for RGB).
        """
        super().__init__()

        # Block 1: 1 conv → 224×224 → 112×112
        self.block1 = nn.Sequential(
            _conv_bn_relu(in_channels, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Block 2: 1 conv → 112×112 → 56×56
        self.block2 = nn.Sequential(
            _conv_bn_relu(64, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Block 3: 2 convs → 56×56 → 28×28
        self.block3 = nn.Sequential(
            _conv_bn_relu(128, 256),
            _conv_bn_relu(256, 256),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Block 4: 2 convs → 28×28 → 14×14
        self.block4 = nn.Sequential(
            _conv_bn_relu(256, 512),
            _conv_bn_relu(512, 512),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Block 5: 2 convs → 14×14 → 7×7  (bottleneck)
        self.block5 = nn.Sequential(
            _conv_bn_relu(512, 512),
            _conv_bn_relu(512, 512),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Args:
            x: Input image tensor [B, in_channels, H, W]. Expects H=W=224.
            return_features: If True, also return skip-connection feature maps.

        Returns:
            If return_features=False:
                bottleneck: [B, 512, 7, 7] feature tensor.
            If return_features=True:
                (bottleneck, features) where features is a dict:
                  "s1": [B, 64,  112, 112]
                  "s2": [B, 128,  56,  56]
                  "s3": [B, 256,  28,  28]
                  "s4": [B, 512,  14,  14]
        """
        s1 = self.block1(x)   # [B,  64, 112, 112]
        s2 = self.block2(s1)  # [B, 128,  56,  56]
        s3 = self.block3(s2)  # [B, 256,  28,  28]
        s4 = self.block4(s3)  # [B, 512,  14,  14]
        s5 = self.block5(s4)  # [B, 512,   7,   7]  ← bottleneck

        if return_features:
            return s5, {"s1": s1, "s2": s2, "s3": s3, "s4": s4}
        return s5


VGG11 = VGG11Encoder


# """VGG11 encoder
# """

# from typing import Dict, Tuple, Union

# import torch
# import torch.nn as nn


# class VGG11Encoder(nn.Module):
#     """VGG11-style encoder with optional intermediate feature returns.
#     """

#     def __init__(self, in_channels: int = 3):
#         """Initialize the VGG11Encoder model."""
#         pass

#     def forward(
#         self, x: torch.Tensor, return_features: bool = False
#     ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
#         """Forward pass.

#         Args:
#             x: input image tensor [B, 3, H, W].
#             return_features: if True, also return skip maps for U-Net decoder.

#         Returns:
#             - if return_features=False: bottleneck feature tensor.
#             - if return_features=True: (bottleneck, feature_dict).
#         """
#         # TODO: Implement forward pass.
#         raise NotImplementedError("Implement VGG11Encoder.forward")