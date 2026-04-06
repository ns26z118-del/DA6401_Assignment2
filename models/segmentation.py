"""U-Net style semantic segmentation using VGG11 as the encoder.

Architecture:
    Encoder (contracting path) — VGG11Encoder blocks 1-5
    Decoder (expansive path)   — symmetric upsampling with skip connections

Upsampling:
    ConvTranspose2d (learnable transposed convolution) is used exclusively
    for upsampling. Bilinear interpolation and unpooling are not used, as
    required by the assignment.

Skip connections:
    At each decoder stage, the upsampled feature map is CONCATENATED with
    the corresponding encoder feature map (same spatial size) along the
    channel dimension before the next conv block. This preserves fine-grained
    spatial details lost during downsampling.

Loss function justification:
    We use CrossEntropyLoss for the 3-class trimap segmentation (pet=0,
    background=1, border=2). CE loss is appropriate because:
    1. The trimap has 3 mutually exclusive classes per pixel.
    2. CE with class weights handles the class imbalance between foreground
       and background pixels.
    Alternative: Dice loss directly optimises the Dice metric, but CE+Dice
    combined tends to be most stable in practice.

Decoder channel sizes (symmetric to encoder):
    Block 5 bottleneck: 512 ch, 7×7
    Up1: ConvTranspose(512→512) + cat(s4=512) → Conv(1024→512), 14×14
    Up2: ConvTranspose(512→256) + cat(s3=256) → Conv(512→256),  28×28
    Up3: ConvTranspose(256→128) + cat(s2=128) → Conv(256→128),  56×56
    Up4: ConvTranspose(128→64)  + cat(s1=64)  → Conv(128→64),  112×112
    Up5: ConvTranspose(64→32)                 → Conv(32→num_classes), 224×224
"""

import torch
import torch.nn as nn

from .vgg11 import VGG11Encoder
from .layers import CustomDropout


def _dec_conv_block(in_c: int, out_c: int) -> nn.Sequential:
    """Decoder conv block: Conv → BN → ReLU → Conv → BN → ReLU."""
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    )


class VGG11UNet(nn.Module):
    """U-Net style segmentation: VGG11 encoder + symmetric transposed-conv decoder."""

    def __init__(self, num_classes: int = 3, in_channels: int = 3, dropout_p: float = 0.5):
        """
        Args:
            num_classes: Number of segmentation classes (3 for pet/bg/border).
            in_channels: Number of input channels.
            dropout_p:   Dropout probability (applied in bottleneck).
        """
        super().__init__()
        self.encoder = VGG11Encoder(in_channels=in_channels)
        self.dropout = CustomDropout(p=dropout_p)

        # ── Decoder ───────────────────────────────────────────────────────────
        # Each stage: ConvTranspose2d (upsample 2×) → cat skip → conv block

        # Up1: 7×7 → 14×14  |  in=512, after cat with s4(512) → 1024
        self.up1    = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.dec1   = _dec_conv_block(512 + 512, 512)

        # Up2: 14×14 → 28×28  |  in=512, after cat with s3(256) → 768
        self.up2    = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec2   = _dec_conv_block(256 + 256, 256)

        # Up3: 28×28 → 56×56  |  in=256, after cat with s2(128) → 384
        self.up3    = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3   = _dec_conv_block(128 + 128, 128)

        # Up4: 56×56 → 112×112  |  in=128, after cat with s1(64) → 192
        self.up4    = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec4   = _dec_conv_block(64 + 64, 64)

        # Up5: 112×112 → 224×224  |  no skip at this stage
        self.up5    = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec5   = _dec_conv_block(32, 32)

        # Final 1×1 conv to produce per-pixel class logits
        self.head   = nn.Conv2d(32, num_classes, kernel_size=1)

    def load_encoder_weights(self, classifier_path: str):
        """Load encoder weights from a trained VGG11Classifier checkpoint."""
        state = torch.load(classifier_path, map_location="cpu")
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
            logits: [B, num_classes, H, W] segmentation logits.
        """
        # ── Encoder (with skip connections) ───────────────────────────────────
        bottleneck, skips = self.encoder(x, return_features=True)
        # bottleneck: [B, 512,  7,  7]
        # skips: s1=[B,64,112,112], s2=[B,128,56,56], s3=[B,256,28,28], s4=[B,512,14,14]

        z = self.dropout(bottleneck)

        # ── Decoder ───────────────────────────────────────────────────────────
        z = self.up1(z)                              # [B, 512, 14, 14]
        z = torch.cat([z, skips["s4"]], dim=1)       # [B, 1024, 14, 14]
        z = self.dec1(z)                             # [B, 512, 14, 14]

        z = self.up2(z)                              # [B, 256, 28, 28]
        z = torch.cat([z, skips["s3"]], dim=1)       # [B, 512, 28, 28]
        z = self.dec2(z)                             # [B, 256, 28, 28]

        z = self.up3(z)                              # [B, 128, 56, 56]
        z = torch.cat([z, skips["s2"]], dim=1)       # [B, 256, 56, 56]
        z = self.dec3(z)                             # [B, 128, 56, 56]

        z = self.up4(z)                              # [B, 64, 112, 112]
        z = torch.cat([z, skips["s1"]], dim=1)       # [B, 128, 112, 112]
        z = self.dec4(z)                             # [B, 64, 112, 112]

        z = self.up5(z)                              # [B, 32, 224, 224]
        z = self.dec5(z)                             # [B, 32, 224, 224]

        return self.head(z)                          # [B, num_classes, 224, 224]

# """Segmentation model
# """

# import torch
# import torch.nn as nn

# class VGG11UNet(nn.Module):
#     """U-Net style segmentation network.
#     """

#     def __init__(self, num_classes: int = 3, in_channels: int = 3, dropout_p: float = 0.5):
#         """
#         Initialize the VGG11UNet model.

#         Args:
#             num_classes: Number of output classes.
#             in_channels: Number of input channels.
#             dropout_p: Dropout probability for the segmentation head.
#         """
#         pass

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """Forward pass for segmentation model.
#         Args:
#             x: Input tensor of shape [B, in_channels, H, W].

#         Returns:
#             Segmentation logits [B, num_classes, H, W].
#         """
#         # TODO: Implement forward pass.
#         raise NotImplementedError("Implement VGG11UNet.forward")
