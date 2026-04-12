 
import torch
import torch.nn as nn

from .vgg11 import VGG11Encoder
from .layers import CustomDropout


class VGG11Classifier(nn.Module):
 

    def __init__(self, num_classes: int = 37, in_channels: int = 3, dropout_p: float = 0.5):
 
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
 