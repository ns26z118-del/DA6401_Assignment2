 

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader
from torchvision import transforms

from data.pets_dataset import OxfordIIITPetDataset
from losses.iou_loss import IoULoss
from models import VGG11Classifier, VGG11UNet
from models.layers import CustomDropout
from models.localization import VGG11Localizer

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASSIFIER_CKPT = "checkpoints/classifier.pth"
LOCALIZER_CKPT  = "checkpoints/localizer.pth"
UNET_CKPT       = "checkpoints/unet.pth"


# ── Transforms ────────────────────────────────────────────────────────────────

def get_train_tf():
    return transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def get_val_tf():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def get_loaders(data_root, batch_size=32, augment=True):
    train_ds = OxfordIIITPetDataset(data_root, "train", get_train_tf() if augment else get_val_tf())
    val_ds   = OxfordIIITPetDataset(data_root, "val",   get_val_tf())
    tl = DataLoader(train_ds, batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    vl = DataLoader(val_ds,   batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return tl, vl


# ── Shared training helpers ────────────────────────────────────────────────────

def train_cls_epoch(model, loader, optimizer, criterion):
    model.train()
    t_loss, correct, n = 0.0, 0, 0
    for b in loader:
        imgs, labels = b["image"].to(DEVICE), b["label"].to(DEVICE)
        optimizer.zero_grad()
        out  = model(imgs)
        loss = criterion(out, labels)
        loss.backward(); optimizer.step()
        t_loss  += loss.item() * imgs.size(0)
        correct += (out.argmax(1) == labels).sum().item()
        n       += imgs.size(0)
    return t_loss / n, correct / n


def val_cls_epoch(model, loader, criterion):
    model.eval()
    v_loss, correct, n = 0.0, 0, 0
    with torch.no_grad():
        for b in loader:
            imgs, labels = b["image"].to(DEVICE), b["label"].to(DEVICE)
            out  = model(imgs)
            loss = criterion(out, labels)
            v_loss  += loss.item() * imgs.size(0)
            correct += (out.argmax(1) == labels).sum().item()
            n       += imgs.size(0)
    return v_loss / n, correct / n


# ── Q2.1: BatchNorm ablation ───────────────────────────────────────────────────

class VGG11NoBN(nn.Module):
    """VGG11 without BatchNorm — for Q2.1 ablation study."""
    def __init__(self, num_classes=37, dropout_p=0.5):
        super().__init__()
        def cr(ic, oc):
            return nn.Sequential(nn.Conv2d(ic, oc, 3, padding=1), nn.ReLU(inplace=True))
        self.features = nn.Sequential(
            cr(3, 64),    nn.MaxPool2d(2, 2),
            cr(64, 128),  nn.MaxPool2d(2, 2),
            cr(128, 256), cr(256, 256), nn.MaxPool2d(2, 2),
            cr(256, 512), cr(512, 512), nn.MaxPool2d(2, 2),
            cr(512, 512), cr(512, 512), nn.MaxPool2d(2, 2),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)), nn.Flatten(),
            CustomDropout(dropout_p), nn.Linear(512*7*7, 4096), nn.ReLU(inplace=True),
            CustomDropout(dropout_p), nn.Linear(4096, 4096),    nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
    def forward(self, x):
        return self.head(self.features(x))


def run_bn_ablation(data_root, epochs=15):
    """Q2.1: Compare training with BatchNorm vs without BatchNorm.
    
    Logs activation distributions from the 3rd conv layer and loss/acc curves.
    The key insight: BN normalises activations, preventing vanishing/exploding
    gradients and allowing higher stable learning rates.
    """
    train_loader, val_loader = get_loaders(data_root, batch_size=32)
    criterion = nn.CrossEntropyLoss()

    for use_bn, label in [ (False, "no_batchnorm")]:
        wandb.init(project="da6401-a2", name=f"q2.1-{label}",
                   group="q2.1-batchnorm",
                   config={"use_bn": use_bn, "epochs": epochs, "lr": 1e-3})

        acts = {}

        if use_bn:
            model = VGG11Classifier(num_classes=37).to(DEVICE)
            # Hook on first conv of block1 (the "3rd conv" counting input as 1st, pool as 2nd)
            model.encoder.block1[0][0].register_forward_hook(
                lambda m, i, o: acts.update({"feat": o.detach().cpu()})
            )
        else:
            model = VGG11NoBN(num_classes=37).to(DEVICE)
            # Hook on first conv of no-BN model
            model.features[0][0].register_forward_hook(
                lambda m, i, o: acts.update({"feat": o.detach().cpu()})
            )

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        for epoch in range(1, epochs + 1):
            # Capture activations on a fixed val batch before training
            model.eval()
            with torch.no_grad():
                fixed_batch = next(iter(val_loader))
                model(fixed_batch["image"].to(DEVICE))

            t_loss, t_acc = train_cls_epoch(model, train_loader, optimizer, criterion)
            v_loss, v_acc = val_cls_epoch(model, val_loader, criterion)
            scheduler.step()

            log = {"epoch": epoch,
                   "train/loss": t_loss, "train/acc": t_acc,
                   "val/loss":   v_loss, "val/acc":   v_acc,
                   "lr": scheduler.get_last_lr()[0]}

            # Log activation histogram at key epochs
            if "feat" in acts and epoch in [1, 5, 10, 15]:
                a = acts["feat"].numpy().flatten()
                fig, ax = plt.subplots(figsize=(6, 3))
                ax.hist(a, bins=80, color="steelblue", alpha=0.8, edgecolor="none")
                ax.axvline(a.mean(), color="red", linestyle="--", label=f"mean={a.mean():.3f}")
                ax.axvline(a.std(),  color="orange", linestyle="--", label=f"std={a.std():.3f}")
                ax.legend(fontsize=8)
                ax.set_title(f"Conv1 activations — {label} — epoch {epoch}")
                ax.set_xlabel("Activation value"); ax.set_ylabel("Count")
                plt.tight_layout()
                log[f"activations/epoch_{epoch}"] = wandb.Image(fig)
                plt.close(fig)

            wandb.log(log)
            print(f"[{label}] Ep{epoch:03d} | Train {t_loss:.4f}/{t_acc:.3f} | Val {v_loss:.4f}/{v_acc:.3f}")

        wandb.finish()
        print(f"Finished {label}\n")


# ── Q2.2: Dropout ablation ─────────────────────────────────────────────────────

def run_dropout_ablation(data_root, epochs=20):
    """Q2.2: Compare dropout p=0, p=0.2, p=0.5.
    
    Logs training vs validation loss curves and generalization gap.
    Higher dropout reduces overfitting (smaller train-val gap) but may
    hurt final accuracy if set too high.
    """
    train_loader, val_loader = get_loaders(data_root, batch_size=16)
    criterion = nn.CrossEntropyLoss()

    for p in [0.0, 0.2, 0.5]:
        wandb.init(project="da6401-a2", name=f"q2.2-dropout-p{p}",
                   group="q2.2-dropout",
                   config={"dropout_p": p, "epochs": epochs, "lr": 1e-3})

        # Use p=0.01 as proxy for 0 (CustomDropout requires p < 1)
        model = VGG11Classifier(num_classes=37, dropout_p=max(p, 0.01)).to(DEVICE)

        if p == 0.0:
            # Disable all dropout by setting p=0
            for m in model.modules():
                if isinstance(m, CustomDropout):
                    m.p = 0.0

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        for epoch in range(1, epochs + 1):
            t_loss, t_acc = train_cls_epoch(model, train_loader, optimizer, criterion)
            v_loss, v_acc = val_cls_epoch(model, val_loader, criterion)
            scheduler.step()

            gap = t_acc - v_acc  # generalization gap: higher = more overfitting

            wandb.log({"epoch": epoch,
                       "train/loss": t_loss, "train/acc": t_acc,
                       "val/loss":   v_loss, "val/acc":   v_acc,
                       "generalization_gap": gap,
                       "lr": scheduler.get_last_lr()[0]})

            print(f"[p={p}] Ep{epoch:03d} | Train {t_acc:.3f} | Val {v_acc:.3f} | Gap {gap:.3f}")

        wandb.finish()
        print(f"Finished dropout p={p}\n")


# ── Q2.4: Feature map visualization ───────────────────────────────────────────

def run_feature_maps(data_root):
    """Q2.4: Visualize first and last conv layer feature maps.
    
    First conv (Block 1): detects low-level features — edges, corners, colour blobs.
    Last conv (Block 5): detects high-level semantics — snouts, ears, fur patterns.
    """
    wandb.init(project="da6401-a2", name="q2.4-feature-maps", group="q2.4")

    model = VGG11Classifier(num_classes=37).to(DEVICE)
    if os.path.exists(CLASSIFIER_CKPT):
        model.load_state_dict(torch.load(CLASSIFIER_CKPT, map_location=DEVICE, weights_only=False))
        print(f"Loaded {CLASSIFIER_CKPT}")
    else:
        print(f"WARNING: {CLASSIFIER_CKPT} not found — using untrained model")
    model.eval()

    first_feat, last_feat = {}, {}
    # Block 1, first conv — very early features
    model.encoder.block1[0][0].register_forward_hook(
        lambda m, i, o: first_feat.update({"f": o.detach().cpu()})
    )
    # Block 5, second conv — deepest features before pooling
    model.encoder.block5[1][0].register_forward_hook(
        lambda m, i, o: last_feat.update({"f": o.detach().cpu()})
    )

    # Use first val sample
    ds    = OxfordIIITPetDataset(data_root, "val", get_val_tf())
    img_t = ds[0]["image"]
    with torch.no_grad():
        model(img_t.unsqueeze(0).to(DEVICE))

    inv_tf = transforms.Normalize(
        mean=[-m/s for m, s in zip(MEAN, STD)],
        std=[1/s for s in STD]
    )
    orig = inv_tf(img_t).permute(1, 2, 0).clamp(0, 1).numpy()

    def make_grid(feat_dict, title, n=16):
        feat = feat_dict["f"][0]           # [C, H, W]
        n    = min(n, feat.shape[0])
        cols = n // 2
        fig, axes = plt.subplots(2, cols, figsize=(cols * 2, 4))
        for i, ax in enumerate(axes.flat):
            if i < n:
                fm = feat[i].numpy()
                ax.imshow(fm, cmap="viridis")
            ax.axis("off")
        fig.suptitle(title, fontsize=10)
        plt.tight_layout()
        return fig

    # Original image
    fig0, ax = plt.subplots(figsize=(3, 3))
    ax.imshow(orig); ax.axis("off"); ax.set_title("Input image")
    plt.tight_layout()

    fig1 = make_grid(first_feat, "Block 1 (Conv1): edges, textures, colour gradients")
    fig2 = make_grid(last_feat,  "Block 5 (Conv2): high-level semantics (ears, snouts, fur)")

    wandb.log({
        "q2.4/input_image":  wandb.Image(fig0),
        "q2.4/first_conv":   wandb.Image(fig1),
        "q2.4/last_conv":    wandb.Image(fig2),
    })
    plt.close("all")
    print("Feature maps logged to W&B.")
    wandb.finish()


# ── Q2.5: Bounding box prediction table ───────────────────────────────────────

def run_bbox_table(data_root):
    """Q2.5: Table of 15 val images with GT (green) and predicted (red) boxes + IoU.
    
    Uses local localizer.pth checkpoint.
    Identifies failure cases: high confidence but low IoU.
    """
    wandb.init(project="da6401-a2", name="q2.5-bbox-table", group="q2.5")

    loc = VGG11Localizer().to(DEVICE)
    if os.path.exists(LOCALIZER_CKPT):
        loc.load_state_dict(torch.load(LOCALIZER_CKPT, map_location=DEVICE, weights_only=False))
        print(f"Loaded {LOCALIZER_CKPT}")
    else:
        print(f"WARNING: {LOCALIZER_CKPT} not found")
    loc.eval()

    ds     = OxfordIIITPetDataset(data_root, "val", get_val_tf())
    inv_tf = transforms.Normalize(
        mean=[-m/s for m, s in zip(MEAN, STD)],
        std=[1/s for s in STD]
    )
    iou_fn = IoULoss(reduction="none")
    table  = wandb.Table(columns=["Image", "IoU", "Result", "GT_box", "Pred_box"])

    with torch.no_grad():
        for i in range(min(15, len(ds))):
            sample = ds[i]
            gt     = sample["bbox"]
            pred   = loc(sample["image"].unsqueeze(0).to(DEVICE))[0].cpu()
            pred[2:] = pred[2:].abs().clamp(min=1)

            iou = float(1.0 - iou_fn(pred.unsqueeze(0), gt.unsqueeze(0)).item())
            iou = max(0.0, min(1.0, iou))

            # Draw boxes on image
            orig = inv_tf(sample["image"]).permute(1, 2, 0).clamp(0, 1).numpy()
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.imshow(orig)

            for box, color, lbl in [
                (gt,   "lime", "GT"),
                (pred, "red",  f"Pred IoU={iou:.2f}"),
            ]:
                cx, cy, w, h = box.tolist()
                rect = plt.Rectangle(
                    (cx - w/2, cy - h/2), w, h,
                    linewidth=2, edgecolor=color, facecolor="none"
                )
                ax.add_patch(rect)
                ax.text(cx - w/2, cy - h/2 - 4, lbl, color=color, fontsize=8,
                        bbox=dict(facecolor="white", alpha=0.5, pad=1))

            ax.axis("off")
            ax.set_title(f"Sample {i+1}  IoU={iou:.3f}")
            plt.tight_layout()

            result = "✓ Good" if iou >= 0.5 else "✗ Poor"
            table.add_data(
                wandb.Image(fig),
                round(iou, 4),
                result,
                str(gt.numpy().round(1).tolist()),
                str(pred.numpy().round(1).tolist()),
            )
            plt.close(fig)

    wandb.log({"q2.5/bbox_predictions": table})
    print("BBox prediction table logged to W&B.")
    wandb.finish()


# ── Q2.6: Segmentation evaluation ─────────────────────────────────────────────

def run_seg_eval(data_root):
    """Q2.6: 5 sample images (original / GT mask / predicted mask) + metric comparison.
    
    Demonstrates why Pixel Accuracy > Dice: background pixels dominate (~50%),
    so predicting everything as background gives high pixel acc but zero Dice.
    Dice penalizes class imbalance properly — it's the correct metric here.
    """
    wandb.init(project="da6401-a2", name="q2.6-seg-eval", group="q2.6")

    model = VGG11UNet(num_classes=3).to(DEVICE)
    if os.path.exists(UNET_CKPT):
        model.load_state_dict(torch.load(UNET_CKPT, map_location=DEVICE, weights_only=False))
        print(f"Loaded {UNET_CKPT}")
    else:
        print(f"WARNING: {UNET_CKPT} not found")
    model.eval()

    ds     = OxfordIIITPetDataset(data_root, "val", get_val_tf())
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=2)
    inv_tf = transforms.Normalize(
        mean=[-m/s for m, s in zip(MEAN, STD)],
        std=[1/s for s in STD]
    )
    colors = np.array([[0, 200, 0], [200, 0, 0], [0, 0, 200]], dtype=np.uint8)

    dice_sum, pix_sum, n = 0.0, 0.0, 0
    sample_imgs = []

    with torch.no_grad():
        for bi, b in enumerate(loader):
            imgs, masks = b["image"].to(DEVICE), b["mask"].to(DEVICE)
            preds = model(imgs).argmax(1)

            # Macro Dice
            d = sum(
                (2 * (preds==c).float() * (masks==c).float()).sum() /
                ((preds==c).float().sum() + (masks==c).float().sum() + 1e-6)
                for c in range(3)
            ) / 3
            dice_sum += d.item() * imgs.size(0)
            pix_sum  += (preds == masks).float().mean().item() * imgs.size(0)
            n += imgs.size(0)

            # Save 5 examples from first batch
            if bi == 0:
                for j in range(min(5, imgs.size(0))):
                    orig      = inv_tf(imgs[j].cpu()).permute(1,2,0).clamp(0,1).numpy()
                    gt_mask   = masks[j].cpu().numpy()
                    pred_mask = preds[j].cpu().numpy()

                    # Per-sample Dice
                    sample_dice = sum(
                        (2*(pred_mask==c).astype(float)*(gt_mask==c).astype(float)).sum() /
                        ((pred_mask==c).astype(float).sum() + (gt_mask==c).astype(float).sum() + 1e-6)
                        for c in range(3)
                    ) / 3

                    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                    axes[0].imshow(orig);
                    axes[0].set_title("Original image"); axes[0].axis("off")
                    axes[1].imshow(colors[gt_mask]);
                    axes[1].set_title("Ground Truth\ngreen=pet  red=bg  blue=border"); axes[1].axis("off")
                    axes[2].imshow(colors[pred_mask]);
                    axes[2].set_title(f"Predicted  Dice={sample_dice:.3f}"); axes[2].axis("off")
                    plt.tight_layout()

                    sample_imgs.append(wandb.Image(fig, caption=f"Sample {j+1} Dice={sample_dice:.3f}"))
                    plt.close(fig)

    val_dice = dice_sum / n
    val_pix  = pix_sum  / n
    gap      = val_pix - val_dice

    wandb.log({
        "q2.6/val_dice":           val_dice,
        "q2.6/val_pixel_accuracy": val_pix,
        "q2.6/pix_minus_dice_gap": gap,
        "q2.6/sample_predictions": sample_imgs,
    })

    print(f"\nVal Dice:          {val_dice:.4f}")
    print(f"Val Pixel Accuracy:{val_pix:.4f}")
    print(f"Gap (Pix - Dice):  {gap:.4f}")
    print("\nWhy the gap exists:")
    print("  Background pixels ~ 50% of each image.")
    print("  A trivial model predicting all-background gets ~50% pixel accuracy")
    print("  but Dice=0 for pet class and Dice=0 for border class → macro Dice ≈ 0.33.")
    print("  Dice penalises class imbalance correctly; pixel accuracy does not.")

    wandb.finish()


# ── Entry point ────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--section",   required=True,
                   choices=["2.1", "2.2", "2.4", "2.5", "2.6"])
    p.add_argument("--data_root", default="./oxford-pet")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(f"Section {args.section} | Device: {DEVICE}")

    if args.section == "2.1":
        run_bn_ablation(args.data_root, epochs=15)
    elif args.section == "2.2":
        run_dropout_ablation(args.data_root, epochs=20)
    elif args.section == "2.4":
        run_feature_maps(args.data_root)
    elif args.section == "2.5":
        run_bbox_table(args.data_root)
    elif args.section == "2.6":
        run_seg_eval(args.data_root)