"""Unified multi-task perception model."""

import torch
import torch.nn as nn

from .classification import VGG11Classifier
from .localization import VGG11Localizer
from .segmentation import VGG11UNet


class MultiTaskPerceptionModel(nn.Module):

    def __init__(
        self,
        num_breeds: int = 37,
        seg_classes: int = 3,
        in_channels: int = 3,
        classifier_path: str = "classifier.pth",
        localizer_path: str = "localizer.pth",
        unet_path: str = "unet.pth",
    ):
        super().__init__()

        import gdown
        gdown.download(id="1vEbZVBJvdubwkaNuQuWdwr6eQE5XOLZB", output=classifier_path, quiet=False)
        gdown.download(id="16nuWdMb3seTLSvaHDuIlVS4CEe4HaylQ", output=localizer_path, quiet=False)
        gdown.download(id="1MEPQv-Pin7ZP45ryTi9LrXfS9jAhlP1S",      output=unet_path,      quiet=False)

        self.classifier = VGG11Classifier(num_classes=num_breeds, in_channels=in_channels)
        self.localizer  = VGG11Localizer(in_channels=in_channels)
        self.segmenter  = VGG11UNet(num_classes=seg_classes, in_channels=in_channels)

        self.classifier.load_state_dict(torch.load(classifier_path, map_location="cpu", weights_only=False))
        self.localizer.load_state_dict( torch.load(localizer_path,  map_location="cpu", weights_only=False))
        self.segmenter.load_state_dict( torch.load(unet_path,       map_location="cpu", weights_only=False))

        self.eval()

    def forward(self, x: torch.Tensor) -> dict:
        self.classifier.eval()
        self.localizer.eval()
        self.segmenter.eval()

        with torch.no_grad():
            cls_out = self.classifier(x)

            boxes = self.localizer(x)
            # Clamp to valid pixel range and ensure w,h are positive
            boxes[:, 0] = boxes[:, 0].clamp(0, 224)   # cx
            boxes[:, 1] = boxes[:, 1].clamp(0, 224)   # cy
            boxes[:, 2] = boxes[:, 2].abs().clamp(1, 224)  # w
            boxes[:, 3] = boxes[:, 3].abs().clamp(1, 224)  # h

            seg_out = self.segmenter(x)

        return {
            "classification": cls_out,
            "localization":   boxes,
            "segmentation":   seg_out,
        }