"""VGG11-based object localization model.

Output: (cx, cy, w, h) in PIXEL space (0-224). No Sigmoid — raw linear output.
Training loss: MSELoss + IoULoss (both operate in pixel space).

Encoder strategy: freeze encoder, only train regression head.
This is faster on limited hardware and works well since the encoder
already learned strong visual features from classification.
"""

import torch
import torch.nn as nn

from .vgg11 import VGG11Encoder
from .layers import CustomDropout


class VGG11Localizer(nn.Module):

    def __init__(self, in_channels: int = 3, dropout_p: float = 0.5):
        super().__init__()
        self.encoder = VGG11Encoder(in_channels=in_channels)

        self.regressor = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            CustomDropout(p=dropout_p),
            nn.Linear(512 * 7 * 7, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(4096, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 4),
            # NO Sigmoid — output is raw pixel values
        )

    def load_encoder_weights(self, classifier_path: str):
        state = torch.load(classifier_path, map_location="cpu", weights_only=False)
        encoder_state = {
            k.replace("encoder.", ""): v
            for k, v in state.items()
            if k.startswith("encoder.")
        }
        self.encoder.load_state_dict(encoder_state)
        print(f"Loaded encoder weights from {classifier_path}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns [B, 4] in (cx, cy, w, h) pixel space."""
        return self.regressor(self.encoder(x))