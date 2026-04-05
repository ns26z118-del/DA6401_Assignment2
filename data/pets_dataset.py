"""Oxford-IIIT Pet multi-task dataset loader.

Provides images, breed labels, bounding boxes, and segmentation trimaps
in a single __getitem__ call — suitable for all four tasks in Assignment 2.

Dataset structure expected at `root`:
    root/
      images/              ← .jpg files (e.g. Abyssinian_1.jpg)
      annotations/
        list.txt           ← image_name class_id species breed_id (1-indexed)
        xmls/              ← PASCAL VOC bounding box XMLs
        trimaps/           ← segmentation masks as .png
                             pixel values: 1=pet, 2=background, 3=border

Output bounding boxes:
    Format: (x_center, y_center, width, height) in pixel coordinates of the
    resized image (target_size). This matches the localization model output
    format required by the assignment.

Output masks:
    Values remapped to 0-indexed: 0=pet, 1=background, 2=border.
"""

import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class OxfordIIITPetDataset(Dataset):
    """Oxford-IIIT Pet multi-task dataset loader."""

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_size: Tuple[int, int] = (224, 224),
    ):
        """
        Args:
            root:        Path to dataset root (contains images/ and annotations/).
            split:       'train', 'val', or 'test'.
                         Uses 90/10 split on the full list (no official train split).
            transform:   torchvision transforms applied to the PIL image.
                         Should include ToTensor() and Normalize().
            target_size: (H, W) to resize images and masks to. Fixed at 224×224
                         to match VGG11 input requirements.
        """
        super().__init__()
        self.root = Path(root)
        self.transform = transform
        self.target_size = target_size  # (H, W)

        # ── Parse list.txt ────────────────────────────────────────────────────
        list_file = self.root / "annotations" / "list.txt"
        all_entries = []
        with open(list_file, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("#") or not line:
                    continue
                parts = line.split()
                name = parts[0]               # e.g. "Abyssinian_100"
                class_id = int(parts[1]) - 1  # 1-indexed in file → 0-indexed
                all_entries.append((name, class_id))

        # ── Train / val / test split ──────────────────────────────────────────
        n = len(all_entries)
        train_end = int(0.9 * n)

        if split == "train":
            self.entries = all_entries[:train_end]
        elif split in ("val", "test"):
            self.entries = all_entries[train_end:]
        else:
            # "all" → entire dataset (useful for inference)
            self.entries = all_entries

    def __len__(self) -> int:
        return len(self.entries)

    def _load_bbox(self, name: str, orig_w: int, orig_h: int) -> torch.Tensor:
        """Load bounding box from PASCAL VOC XML.

        Scales box coordinates from original image size to target_size,
        then converts (x1,y1,x2,y2) → (cx,cy,w,h) in pixel space.

        Falls back to the full image bounding box if the XML is missing.
        """
        xml_path = self.root / "annotations" / "xmls" / f"{name}.xml"

        if not xml_path.exists():
            # Fallback: entire image is the bounding box
            cx = self.target_size[1] / 2.0
            cy = self.target_size[0] / 2.0
            w  = float(self.target_size[1])
            h  = float(self.target_size[0])
            return torch.tensor([cx, cy, w, h], dtype=torch.float32)

        tree = ET.parse(xml_path)
        root_elem = tree.getroot()
        bndbox = root_elem.find(".//bndbox")

        xmin = float(bndbox.find("xmin").text)
        ymin = float(bndbox.find("ymin").text)
        xmax = float(bndbox.find("xmax").text)
        ymax = float(bndbox.find("ymax").text)

        # Scale to target image dimensions
        scale_x = self.target_size[1] / orig_w
        scale_y = self.target_size[0] / orig_h

        xmin = xmin * scale_x
        xmax = xmax * scale_x
        ymin = ymin * scale_y
        ymax = ymax * scale_y

        # Convert corner format → centre format
        cx = (xmin + xmax) / 2.0
        cy = (ymin + ymax) / 2.0
        w  = xmax - xmin
        h  = ymax - ymin

        return torch.tensor([cx, cy, w, h], dtype=torch.float32)

    def __getitem__(self, idx: int) -> dict:
        """
        Returns a dict with:
            image: Tensor [3, H, W] (after transform) or PIL Image (if no transform)
            label: Tensor scalar — breed class index in [0, 36]
            bbox:  Tensor [4]    — (cx, cy, w, h) in pixel space of target_size image
            mask:  Tensor [H, W] long — 0=pet, 1=background, 2=border
        """
        name, class_id = self.entries[idx]

        # ── Load image ────────────────────────────────────────────────────────
        img_path = self.root / "images" / f"{name}.jpg"
        image = Image.open(img_path).convert("RGB")
        orig_w, orig_h = image.size

        # ── Bounding box ──────────────────────────────────────────────────────
        bbox = self._load_bbox(name, orig_w, orig_h)

        # ── Segmentation mask ─────────────────────────────────────────────────
        mask_path = self.root / "annotations" / "trimaps" / f"{name}.png"
        mask = Image.open(mask_path)
        mask = mask.resize((self.target_size[1], self.target_size[0]), Image.NEAREST)
        # Remap: file values 1,2,3 → 0,1,2
        mask_arr = np.array(mask, dtype=np.int64) - 1
        mask_tensor = torch.from_numpy(mask_arr).long()

        # ── Resize image ──────────────────────────────────────────────────────
        image = image.resize((self.target_size[1], self.target_size[0]), Image.BILINEAR)

        # ── Apply transforms (Normalize, ToTensor, augmentations) ─────────────
        if self.transform is not None:
            image = self.transform(image)

        return {
            "image": image,
            "label": torch.tensor(class_id, dtype=torch.long),
            "bbox":  bbox,
            "mask":  mask_tensor,
        }

# """Dataset skeleton for Oxford-IIIT Pet.
# """

# from torch.utils.data import Dataset

# class OxfordIIITPetDataset(Dataset):
#     """Oxford-IIIT Pet multi-task dataset loader skeleton."""
#     pass