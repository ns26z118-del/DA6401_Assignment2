"""inference_wildcard.py — Q2.7: Run pipeline on 3 internet pet images.

Loads directly from local checkpoint files (no gdown/Drive needed).

Usage:
    1. Save 3 pet images from the internet as pet1.jpg, pet2.jpg, pet3.jpg
       in the project root folder.
    2. python inference_wildcard.py
    3. Results saved as wildcard_1.png, wildcard_2.png, wildcard_3.png
       and logged to W&B under q2.7.
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from PIL import Image
from torchvision import transforms

from models import VGG11Classifier, VGG11Localizer, VGG11UNet

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

# Oxford-IIIT Pet 37 breed names in class index order (0-indexed)
CLASS_NAMES = [
    "Abyssinian", "Bengal", "Birman", "Bombay", "British Shorthair",
    "Egyptian Mau", "Maine Coon", "Persian", "Ragdoll", "Russian Blue",
    "Siamese", "Sphynx", "American Bulldog", "American Pit Bull Terrier",
    "Basset Hound", "Beagle", "Boxer", "Chihuahua", "English Cocker Spaniel",
    "English Setter", "German Shorthaired", "Great Pyrenees", "Havanese",
    "Japanese Chin", "Keeshond", "Leonberger", "Miniature Pinscher",
    "Newfoundland", "Pomeranian", "Pug", "Saint Bernard", "Samoyed",
    "Scottish Terrier", "Shiba Inu", "Staffordshire Bull Terrier",
    "Wheaten Terrier", "Yorkshire Terrier",
]

# Segmentation class colors: 0=pet(green), 1=background(red), 2=border(blue)
SEG_COLORS = np.array([[0, 200, 0], [200, 0, 0], [0, 0, 200]], dtype=np.uint8)

CLASSIFIER_PATH = "checkpoints/classifier.pth"
LOCALIZER_PATH  = "checkpoints/localizer.pth"
UNET_PATH       = "checkpoints/unet.pth"


def load_models(device):
    """Load all three models from local checkpoints."""

    # ── Classifier ────────────────────────────────────────────────────────────
    classifier = VGG11Classifier(num_classes=37).to(device)
    if os.path.exists(CLASSIFIER_PATH):
        classifier.load_state_dict(
            torch.load(CLASSIFIER_PATH, map_location=device, weights_only=False)
        )
        print(f"Loaded classifier from {CLASSIFIER_PATH}")
    else:
        print(f"WARNING: {CLASSIFIER_PATH} not found — using random weights")
    classifier.eval()

    # ── Localizer ─────────────────────────────────────────────────────────────
    localizer = VGG11Localizer().to(device)
    if os.path.exists(LOCALIZER_PATH):
        localizer.load_state_dict(
            torch.load(LOCALIZER_PATH, map_location=device, weights_only=False)
        )
        print(f"Loaded localizer  from {LOCALIZER_PATH}")
    else:
        print(f"WARNING: {LOCALIZER_PATH} not found — using random weights")
    localizer.eval()

    # ── UNet ──────────────────────────────────────────────────────────────────
    segmenter = VGG11UNet(num_classes=3).to(device)
    if os.path.exists(UNET_PATH):
        segmenter.load_state_dict(
            torch.load(UNET_PATH, map_location=device, weights_only=False)
        )
        print(f"Loaded segmenter  from {UNET_PATH}")
    else:
        print(f"WARNING: {UNET_PATH} not found — using random weights")
    segmenter.eval()

    return classifier, localizer, segmenter


def run_inference():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    wandb.init(project="da6401-a2", name="q2.7-wildcard-inference", group="q2.7")

    classifier, localizer, segmenter = load_models(device)

    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])

    image_files = ["pet1.jpg", "pet2.jpg", "pet3.jpg"]
    wandb_images = []
    found = 0

    for i, fname in enumerate(image_files, 1):
        if not os.path.exists(fname):
            print(f"  SKIP: {fname} not found")
            continue

        found += 1
        print(f"\n[{i}] Processing {fname}...")

        img_pil = Image.open(fname).convert("RGB")
        x       = tf(img_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            cls_logits = classifier(x)
            box        = localizer(x)[0].cpu()
            seg_logits = segmenter(x)

        # ── Classification ────────────────────────────────────────────────────
        probs     = torch.softmax(cls_logits, dim=1)[0]
        breed_idx = probs.argmax().item()
        breed     = CLASS_NAMES[breed_idx] if breed_idx < len(CLASS_NAMES) else f"cls_{breed_idx}"
        conf      = probs[breed_idx].item()

        # Top-3 predictions
        top3_vals, top3_idx = probs.topk(3)
        top3 = [(CLASS_NAMES[j.item()] if j.item() < len(CLASS_NAMES) else f"cls_{j.item()}",
                 v.item()) for j, v in zip(top3_idx, top3_vals)]

        # ── Localisation ──────────────────────────────────────────────────────
        box[2:] = box[2:].abs().clamp(min=1)  # ensure positive w, h
        cx, cy, w, h = box.tolist()

        # ── Segmentation ──────────────────────────────────────────────────────
        seg_mask = seg_logits[0].argmax(0).cpu().numpy()
        seg_rgb  = SEG_COLORS[seg_mask]

        # Pet pixel fraction
        pet_pct = (seg_mask == 0).mean() * 100

        # ── Visualise ─────────────────────────────────────────────────────────
        orig_224 = np.array(img_pil.resize((224, 224)))

        fig, axes = plt.subplots(1, 4, figsize=(18, 5))
        fig.suptitle(f"{fname}  →  {breed}  (conf={conf:.2f})", fontsize=12)

        # Panel 1: original
        axes[0].imshow(orig_224)
        axes[0].set_title("Original image")
        axes[0].axis("off")

        # Panel 2: classification top-3
        axes[1].imshow(orig_224, alpha=0.4)
        top3_str = "\n".join([f"{n}: {v:.2f}" for n, v in top3])
        axes[1].text(5, 15, top3_str, color="white", fontsize=9,
                     bbox=dict(facecolor="black", alpha=0.6, pad=4))
        axes[1].set_title(f"Top-3 breeds")
        axes[1].axis("off")

        # Panel 3: bounding box
        axes[2].imshow(orig_224)
        rect = patches.Rectangle(
            (cx - w/2, cy - h/2), w, h,
            linewidth=2, edgecolor="red", facecolor="none"
        )
        axes[2].add_patch(rect)
        axes[2].set_title(f"BBox [cx={cx:.0f} cy={cy:.0f} w={w:.0f} h={h:.0f}]")
        axes[2].axis("off")

        # Panel 4: segmentation overlay
        axes[3].imshow(orig_224, alpha=0.55)
        axes[3].imshow(seg_rgb, alpha=0.45)
        axes[3].set_title(f"Seg: pet={pet_pct:.0f}%\ngreen=pet  red=bg  blue=border")
        axes[3].axis("off")

        plt.tight_layout()
        out_path = f"wildcard_{i}.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"  Saved {out_path}")
        print(f"  Breed: {breed}  conf={conf:.3f}")
        print(f"  BBox:  cx={cx:.1f} cy={cy:.1f} w={w:.1f} h={h:.1f}")
        print(f"  Seg:   pet={pet_pct:.1f}%  bg={(seg_mask==1).mean()*100:.1f}%  border={(seg_mask==2).mean()*100:.1f}%")

        wandb_images.append(
            wandb.Image(fig, caption=f"{fname}: {breed} ({conf:.2f})")
        )
        plt.close(fig)

    if found == 0:
        print("\nNo pet images found!")
        print("Please save pet1.jpg, pet2.jpg, pet3.jpg in the project root.")
    else:
        wandb.log({"q2.7/wildcard_predictions": wandb_images})
        print(f"\nLogged {len(wandb_images)} wildcard predictions to W&B.")

    wandb.finish()


if __name__ == "__main__":
    run_inference()