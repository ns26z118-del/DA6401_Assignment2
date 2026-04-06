"""VGG11-based object localization model.

Encoder adaptation strategy:
    We FINE-TUNE the entire encoder (no freezing) for the following reasons:
    1. The pet head bounding boxes require spatial understanding that differs
       from breed classification — the encoder needs to adapt its feature
       representations to focus on object boundaries rather than texture.
    2. The Oxford-IIIT Pet dataset is small enough that fine-tuning with a
       low learning rate does not cause catastrophic forgetting.
    3. Empirically, fine-tuning consistently outperforms frozen encoders on
       small datasets when the source and target tasks are related but not
       identical (as is the case here: classification vs localisation).

Output format:
    [x_center, y_center, width, height] in pixel coordinates of the
    224×224 input image. Raw linear output (no activation) since pixel
    values are not bounded to [0,1].

Training loss:
    total_loss = MSELoss(pred, target) + IoULoss(pred, target)
    MSE handles direct coordinate regression; IoU directly optimises
    the overlap metric used at evaluation time.
"""

import torch
import torch.nn as nn

from .vgg11 import VGG11Encoder
from .layers import CustomDropout


class VGG11Localizer(nn.Module):
    """VGG11 encoder + regression head for single-object localisation."""

    def __init__(self, in_channels: int = 3, dropout_p: float = 0.5):
        """
        Args:
            in_channels: Number of input channels (3 for RGB).
            dropout_p:   Dropout probability for the regression head.
        """
        super().__init__()
        self.encoder = VGG11Encoder(in_channels=in_channels)

        # Regression head: bottleneck (7×7×512) → 4 bbox coordinates
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

            # Output: 4 values [cx, cy, w, h] in pixel space
            # No activation — raw linear output, pixel values are unbounded
            nn.Linear(1024, 4),
        )

    def load_encoder_weights(self, classifier_path: str):
        """Load encoder weights from a trained VGG11Classifier checkpoint.

        Args:
            classifier_path: Path to classifier.pth saved by train.py.
        """
        state = torch.load(classifier_path, map_location="cpu")
        # Extract only the encoder weights (keys starting with "encoder.")
        encoder_state = {
            k.replace("encoder.", ""): v
            for k, v in state.items()
            if k.startswith("encoder.")
        }
        self.encoder.load_state_dict(encoder_state)
        print(f"Loaded encoder weights from {classifier_path}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, in_channels, H, W] — expects H=W=224, normalised.
        Returns:
            boxes: [B, 4] in (cx, cy, w, h) pixel space.
        """
        features = self.encoder(x)       # [B, 512, 7, 7]
        boxes = self.regressor(features) # [B, 4]
        return boxes

# """Localization modules
# """

# import torch
# import torch.nn as nn

# class VGG11Localizer(nn.Module):
#     """VGG11-based localizer."""

#     def __init__(self, in_channels: int = 3, dropout_p: float = 0.5):
#         """
#         Initialize the VGG11Localizer model.

#         Args:
#             in_channels: Number of input channels.
#             dropout_p: Dropout probability for the localization head.
#         """
#         pass

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """Forward pass for localization model.
#         Args:
#             x: Input tensor of shape [B, in_channels, H, W].

#         Returns:
#             Bounding box coordinates [B, 4] in (x_center, y_center, width, height) format in original image pixel space(not normalized values).
#         """
#         # TODO: Implement forward pass.
#         raise NotImplementedError("Implement VGG11Localizer.forward")
