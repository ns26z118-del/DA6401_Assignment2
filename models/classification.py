"""VGG11-based classification model for 37-breed pet classification."""

import torch
import torch.nn as nn

from .vgg11 import VGG11Encoder
from .layers import CustomDropout


class VGG11Classifier(nn.Module):
    """Full classifier = VGG11Encoder backbone + classification head.

    Head design: AdaptiveAvgPool → Flatten → Dropout → FC(4096) → BN → ReLU
                 → Dropout → FC(4096) → BN → ReLU → FC(num_classes)

    Dropout placement rationale:
      CustomDropout is placed BEFORE each large FC layer. This mirrors the
      original VGG paper and forces the network to develop redundant feature
      representations. Placing BN1d after each FC (before ReLU) stabilises the
      large linear layer activations, enabling higher learning rates and
      faster convergence compared to plain VGG without BN.

    No final activation: raw logits are returned. Use nn.CrossEntropyLoss
    during training (which applies log-softmax internally).
    """

    def __init__(self, num_classes: int = 37, in_channels: int = 3, dropout_p: float = 0.5):
        """
        Args:
            num_classes: Number of output classes (37 for Oxford-IIIT Pet breeds).
            in_channels: Number of input channels (3 for RGB).
            dropout_p:   Dropout probability for CustomDropout layers in head.
        """
        super().__init__()
        self.encoder = VGG11Encoder(in_channels=in_channels)

        # After block5 with 224×224 input: 512 × 7 × 7 = 25088 features
        self.classifier = nn.Sequential(
            # AdaptiveAvgPool ensures correct spatial size regardless of minor
            # input variation — outputs 7×7 per channel
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),

            # FC layer 1
            CustomDropout(p=dropout_p),
            nn.Linear(512 * 7 * 7, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),

            # FC layer 2
            CustomDropout(p=dropout_p),
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),

            # Output layer — no activation (logits)
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, in_channels, H, W] — expects H=W=224, normalised.
        Returns:
            logits: [B, num_classes]
        """
        features = self.encoder(x)         # [B, 512, 7, 7]
        logits = self.classifier(features) # [B, num_classes]
        return logits

# """Classification components
# """

# import torch
# import torch.nn as nn


# class VGG11Classifier(nn.Module):
#     """Full classifier = VGG11Encoder + ClassificationHead."""

#     def __init__(self, num_classes: int = 37, in_channels: int = 3, dropout_p: float = 0.5):
#         """
#         Initialize the VGG11Classifier model.
#         Args:
#             num_classes: Number of output classes.
#             in_channels: Number of input channels.
#             dropout_p: Dropout probability for the classifier head.
#         """
#         pass

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """Forward pass for classification model.
#         Args:
#             x: Input tensor of shape [B, in_channels, H, W].
#         Returns:
#             Classification logits [B, num_classes].
#         """
#         # TODO: Implement forward pass.
#         raise NotImplementedError("Implement VGG11Classifier.forward")
