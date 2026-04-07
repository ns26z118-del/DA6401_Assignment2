"""Unified multi-task model
"""

import torch
import torch.nn as nn

from .classification import VGG11Classifier
from .localization import VGG11Localizer
from .segmentation import VGG11UNet


class MultiTaskPerceptionModel(nn.Module):
    """Shared-backbone multi-task model."""

    def __init__(
        self,
        num_breeds: int = 37,
        seg_classes: int = 3,
        in_channels: int = 3,
        classifier_path: str = "classifier.pth",
        localizer_path: str = "localizer.pth",
        unet_path: str = "unet.pth"
    ):
        super().__init__()

        import gdown
        gdown.download(id="1vEbZVBJvdubwkaNuQuWdwr6eQE5XOLZB", output=classifier_path, quiet=False)
        gdown.download(id="11mpsE7FTREjRf5GTi5uJPsIkXiPnx6L8", output=localizer_path, quiet=False)
        gdown.download(id="1ousFmvDKSxE11A3bD6MPp2IVub_pBwdy", output=unet_path, quiet=False)

        # 🔥 Initialize models
        self.classifier = VGG11Classifier(num_classes=num_breeds, in_channels=in_channels)
        self.localizer = VGG11Localizer(in_channels=in_channels)
        self.segmenter = VGG11UNet(num_classes=seg_classes, in_channels=in_channels)

        # 🔥 Load pretrained weights
        self.classifier.load_state_dict(torch.load(classifier_path, map_location="cpu"))
        self.localizer.load_state_dict(torch.load(localizer_path, map_location="cpu"))
        self.segmenter.load_state_dict(torch.load(unet_path, map_location="cpu"))

        # optional but safe
        self.eval()


 
    def forward(self, x):

        self.classifier.eval()
        self.localizer.eval()
        self.segmenter.eval()

        with torch.no_grad():
            cls_logits = self.classifier(x)

            boxes = self.localizer(x)  # [0,1]

            # ✅ Convert normalized → pixel space properly
            cx = boxes[:, 0] * 224.0
            cy = boxes[:, 1] * 224.0
            w  = boxes[:, 2] * 224.0
            h  = boxes[:, 3] * 224.0

            # ✅ Clamp to valid range
            cx = torch.clamp(cx, 0, 224)
            cy = torch.clamp(cy, 0, 224)
            w  = torch.clamp(w, 1, 224)
            h  = torch.clamp(h, 1, 224)

            boxes = torch.stack([cx, cy, w, h], dim=1)

            seg_logits = self.segmenter(x)

        return {
            "classification": cls_logits,
            "localization": boxes,
            "segmentation": seg_logits
        }

    # def forward(self, x: torch.Tensor):

    #     cls_logits = self.classifier(x)     # [B, num_breeds]
    #     boxes = self.localizer(x)           # [B, 4]
    #     seg_logits = self.segmenter(x)      # [B, seg_classes, H, W]

    #     return {
    #         "classification": cls_logits,
    #         "localization": boxes,
    #         "segmentation": seg_logits
    #     }

# """Unified multi-task model
# """

# import torch
# import torch.nn as nn


# class MultiTaskPerceptionModel(nn.Module):
#     """Shared-backbone multi-task model."""

#     def __init__(self, num_breeds: int = 37, seg_classes: int = 3, in_channels: int = 3, classifier_path: str = "classifier.pth", localizer_path: str = "localizer.pth", unet_path: str = "unet.pth"):
#         """
#         Initialize the shared backbone/heads using these trained weights.
#         Args:
#             num_breeds: Number of output classes for classification head.
#             seg_classes: Number of output classes for segmentation head.
#             in_channels: Number of input channels.
#             classifier_path: Path to trained classifier weights.
#             localizer_path: Path to trained localizer weights.
#             unet_path: Path to trained unet weights.
#         """

#         super().__init__()

#         import gdown
#         gdown.download(id="1vEbZVBJvdubwkaNuQuWdwr6eQE5XOLZB", output=classifier_path, quiet=False)
#         gdown.download(id="1md6XETOyYzGnt7bfzpOrPpswehy1jdDQ", output=localizer_path, quiet=False)
#         gdown.download(id="1EMYz_F6uDZee4Hh4j6MbZGbSSO6Ao79I", output=unet_path, quiet=False)
#         pass

#     def forward(self, x: torch.Tensor):
#         """Forward pass for multi-task model.
#         Args:
#             x: Input tensor of shape [B, in_channels, H, W].
#         Returns:
#             A dict with keys:
#             - 'classification': [B, num_breeds] logits tensor.
#             - 'localization': [B, 4] bounding box tensor.
#             - 'segmentation': [B, seg_classes, H, W] segmentation logits tensor
#         """
#         # TODO: Implement forward pass.
#         raise NotImplementedError("Implement MultiTaskPerceptionModel.forward")
