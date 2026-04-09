"""Oxford-IIIT Pet multi-task dataset loader.

Key design decisions:
- Only includes samples that have bounding box XML annotations
- Bounding boxes are in PIXEL space (0-224), NOT normalized
- Masks remapped from 1,2,3 to 0,1,2
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class OxfordIIITPetDataset(Dataset):

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_size: Tuple[int, int] = (224, 224),
    ):
        super().__init__()
        self.root = Path(root)
        self.transform = transform
        self.target_size = target_size

        list_file = self.root / "annotations" / "list.txt"
        xml_dir   = self.root / "annotations" / "xmls"

        all_entries = []
        with open(list_file, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("#") or not line:
                    continue
                parts    = line.split()
                name     = parts[0]
                class_id = int(parts[1]) - 1  # 0-indexed

                # Only include samples that have an XML bounding box file
                if (xml_dir / f"{name}.xml").exists():
                    all_entries.append((name, class_id))

        print(f"[Dataset] {len(all_entries)} samples with XML annotations")

        n         = len(all_entries)
        split_idx = int(0.9 * n)

        if split == "train":
            self.entries = all_entries[:split_idx]
        elif split in ("val", "test"):
            self.entries = all_entries[split_idx:]
        else:
            self.entries = all_entries

        print(f"[Dataset] {split}: {len(self.entries)} samples")

    def __len__(self):
        return len(self.entries)

    def _load_bbox(self, name: str, orig_w: int, orig_h: int) -> torch.Tensor:
        """Returns (cx, cy, w, h) in PIXEL space of the 224x224 resized image."""
        xml_path  = self.root / "annotations" / "xmls" / f"{name}.xml"
        bndbox    = ET.parse(xml_path).getroot().find(".//bndbox")

        xmin = float(bndbox.find("xmin").text)
        ymin = float(bndbox.find("ymin").text)
        xmax = float(bndbox.find("xmax").text)
        ymax = float(bndbox.find("ymax").text)

        # Scale coordinates to 224x224
        sx = self.target_size[1] / orig_w
        sy = self.target_size[0] / orig_h

        xmin = max(0.0,                     xmin * sx)
        xmax = min(float(self.target_size[1]), xmax * sx)
        ymin = max(0.0,                     ymin * sy)
        ymax = min(float(self.target_size[0]), ymax * sy)

        cx = (xmin + xmax) / 2.0
        cy = (ymin + ymax) / 2.0
        w  = xmax - xmin
        h  = ymax - ymin

        return torch.tensor([cx, cy, w, h], dtype=torch.float32)

    def __getitem__(self, idx: int) -> dict:
        name, class_id = self.entries[idx]

        img   = Image.open(self.root / "images" / f"{name}.jpg").convert("RGB")
        orig_w, orig_h = img.size

        bbox = self._load_bbox(name, orig_w, orig_h)

        mask = Image.open(self.root / "annotations" / "trimaps" / f"{name}.png")
        mask = mask.resize((self.target_size[1], self.target_size[0]), Image.NEAREST)
        mask = torch.from_numpy(np.array(mask, dtype=np.int64) - 1).long()

        img = img.resize((self.target_size[1], self.target_size[0]), Image.BILINEAR)

        if self.transform is not None:
            img = self.transform(img)

        return {
            "image": img,
            "label": torch.tensor(class_id, dtype=torch.long),
            "bbox":  bbox,   # [cx, cy, w, h] pixel space, values ~0-224
            "mask":  mask,   # [H, W] values in {0,1,2}
        }