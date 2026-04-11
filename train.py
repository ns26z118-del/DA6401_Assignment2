"""train.py — DA6401 Assignment 2 training script.

Usage:
    python train.py --task classification --data_root ./oxford-pet --epochs 50
    python train.py --task localization   --data_root ./oxford-pet --epochs 30
    python train.py --task segmentation   --data_root ./oxford-pet --epochs 40 --lr 5e-4
"""

import argparse
import os

import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader
from torchvision import transforms

from data.pets_dataset import OxfordIIITPetDataset
from losses.iou_loss import IoULoss
from models import VGG11Classifier, VGG11Localizer, VGG11UNet

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]


def get_transforms(split: str):
    if split == "train":
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
        ])
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])


def get_plain_transforms():
    """No augmentation — used for localization to keep bbox coords valid."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])


# ── Segmentation loss helpers ──────────────────────────────────────────────────

def dice_loss_fn(logits, masks, num_classes=3):
    """Soft Dice loss on softmax probabilities — differentiable."""
    probs = torch.softmax(logits, dim=1)
    loss  = 0.0
    for c in range(num_classes):
        p = probs[:, c]
        t = (masks == c).float()
        loss += 1.0 - (2.0 * (p * t).sum() + 1e-6) / (p.sum() + t.sum() + 1e-6)
    return loss / num_classes


def compute_dice_score(preds, masks, num_classes=3):
    """Hard Dice score on argmax predictions — for logging only."""
    score = 0.0
    for c in range(num_classes):
        p = (preds == c).float()
        t = (masks == c).float()
        score += (2.0 * (p * t).sum() + 1e-6) / (p.sum() + t.sum() + 1e-6)
    return score / num_classes


def compute_pixel_acc(preds, masks):
    return (preds == masks).float().mean()


# ── Classification ─────────────────────────────────────────────────────────────

def train_classifier(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training classifier on: {device}")
    wandb.init(project="da6401-a2", name=f"cls-drop{args.dropout_p}", config=vars(args))

    train_ds = OxfordIIITPetDataset(args.data_root, "train", get_transforms("train"))
    val_ds   = OxfordIIITPetDataset(args.data_root, "val",   get_transforms("val"))
    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model     = VGG11Classifier(num_classes=37, dropout_p=args.dropout_p).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t_loss, correct, total = 0.0, 0, 0
        for b in train_loader:
            imgs, labels = b["image"].to(device), b["label"].to(device)
            optimizer.zero_grad()
            out  = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            t_loss  += loss.item() * imgs.size(0)
            correct += (out.argmax(1) == labels).sum().item()
            total   += imgs.size(0)
        t_loss /= total
        t_acc   = correct / total

        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for b in val_loader:
                imgs, labels = b["image"].to(device), b["label"].to(device)
                out  = model(imgs)
                loss = criterion(out, labels)
                v_loss    += loss.item() * imgs.size(0)
                v_correct += (out.argmax(1) == labels).sum().item()
                v_total   += imgs.size(0)
        v_loss /= v_total
        v_acc   = v_correct / v_total
        scheduler.step()

        print(f"Epoch {epoch:03d} | Train loss {t_loss:.4f} acc {t_acc:.4f} | "
              f"Val loss {v_loss:.4f} acc {v_acc:.4f}")
        wandb.log({"epoch": epoch, "train/loss": t_loss, "train/acc": t_acc,
                   "val/loss": v_loss, "val/acc": v_acc,
                   "lr": scheduler.get_last_lr()[0]})

        if v_acc > best_acc:
            best_acc = v_acc
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), "checkpoints/classifier.pth")
            print(f"  ✓ Saved (val_acc={v_acc:.4f})")

    wandb.finish()
    print(f"Best val accuracy: {best_acc:.4f}")


# ── Localization ───────────────────────────────────────────────────────────────

def train_localizer(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training localizer on: {device}")
    wandb.init(project="da6401-a2", name="localizer", config=vars(args))

    plain = get_plain_transforms()
    train_ds = OxfordIIITPetDataset(args.data_root, "train", plain)
    val_ds   = OxfordIIITPetDataset(args.data_root, "val",   plain)
    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = VGG11Localizer(dropout_p=args.dropout_p).to(device)

    if os.path.exists("checkpoints/classifier.pth"):
        model.load_encoder_weights("checkpoints/classifier.pth")
        for p in model.encoder.parameters():
            p.requires_grad = False
        print("Encoder frozen — only training regression head")
    else:
        print("WARNING: no classifier.pth found")

    mse_loss = nn.MSELoss()
    iou_loss = IoULoss(reduction="mean")

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        t_loss, n = 0.0, 0
        for b in train_loader:
            imgs   = b["image"].to(device)
            bboxes = b["bbox"].to(device)
            optimizer.zero_grad()
            pred = model(imgs)
            loss = mse_loss(pred, bboxes) + iou_loss(pred, bboxes)
            loss.backward()
            optimizer.step()
            t_loss += loss.item() * imgs.size(0)
            n += imgs.size(0)
        t_loss /= n

        model.eval()
        v_loss, iou_sum, vn = 0.0, 0.0, 0
        with torch.no_grad():
            for b in val_loader:
                imgs   = b["image"].to(device)
                bboxes = b["bbox"].to(device)
                pred   = model(imgs)
                v_loss  += (mse_loss(pred, bboxes) + iou_loss(pred, bboxes)).item() * imgs.size(0)
                iou_sum += (1 - iou_loss(pred, bboxes).item()) * imgs.size(0)
                vn += imgs.size(0)
        v_loss   /= vn
        mean_iou  = iou_sum / vn
        scheduler.step()

        print(f"Epoch {epoch:03d} | Train loss {t_loss:.4f} | Val loss {v_loss:.4f} | Val IoU {mean_iou:.4f}")
        wandb.log({"epoch": epoch, "train/loss": t_loss, "val/loss": v_loss,
                   "val/iou": mean_iou, "lr": scheduler.get_last_lr()[0]})

        if v_loss < best_val:
            best_val = v_loss
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), "checkpoints/localizer.pth")
            print(f"  ✓ Saved (val_loss={v_loss:.4f}, IoU={mean_iou:.4f})")

        if mean_iou > 0.70:
            print(f"IoU {mean_iou:.4f} > 0.70 — stopping early")
            break

    wandb.finish()
    print(f"Best val loss: {best_val:.4f}")


# ── Segmentation ───────────────────────────────────────────────────────────────

def train_segmentation(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training segmentation on: {device}")
    wandb.init(project="da6401-a2", name="segmentation-fulltune", config=vars(args))

    train_ds = OxfordIIITPetDataset(args.data_root, "train", get_transforms("train"))
    val_ds   = OxfordIIITPetDataset(args.data_root, "val",   get_transforms("val"))
    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = VGG11UNet(num_classes=3, dropout_p=args.dropout_p).to(device)

    if os.path.exists("checkpoints/classifier.pth"):
        model.load_encoder_weights("checkpoints/classifier.pth")
        print("Loaded encoder weights — full fine-tuning with differential LR")
    else:
        print("WARNING: classifier.pth not found — training from scratch")

    # ce_loss = nn.CrossEntropyLoss()
    class_weights = torch.tensor([1.0, 1.0, 3.0], device=device)
    ce_loss = nn.CrossEntropyLoss(weight=class_weights)

    # Discriminative fine-tuning:
    # Encoder (pretrained) gets 10x lower LR to preserve learned features
    # Decoder (randomly initialised) gets full LR to learn fast
    encoder_params = list(model.encoder.parameters())
    decoder_params = [p for n, p in model.named_parameters() if "encoder" not in n]

    optimizer = torch.optim.Adam([
        {"params": encoder_params, "lr": args.lr * 0.1},
        {"params": decoder_params, "lr": args.lr},
    ], weight_decay=5e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_dice = 0.0

    for epoch in range(1, args.epochs + 1):
        # ── Train ──────────────────────────────────────────────────────────────
        model.train()
        t_loss, n = 0.0, 0

        for b in train_loader:
            imgs  = b["image"].to(device)
            masks = b["mask"].to(device)

            optimizer.zero_grad()
            logits = model(imgs)

            # Combined CE + soft Dice loss
            ce   = ce_loss(logits, masks)
            dice = dice_loss_fn(logits, masks)
            loss = ce + dice

            loss.backward()
            optimizer.step()
            t_loss += loss.item() * imgs.size(0)
            n += imgs.size(0)
        t_loss /= n

        # ── Validate ───────────────────────────────────────────────────────────
        model.eval()
        v_loss, dice_sum, pix_sum, vn = 0.0, 0.0, 0.0, 0

        with torch.no_grad():
            for b in val_loader:
                imgs  = b["image"].to(device)
                masks = b["mask"].to(device)

                logits = model(imgs)
                ce     = ce_loss(logits, masks)
                dice   = dice_loss_fn(logits, masks)
                v_loss += (ce + dice).item() * imgs.size(0)

                preds     = logits.argmax(dim=1)
                dice_sum += compute_dice_score(preds, masks).item() * imgs.size(0)
                pix_sum  += compute_pixel_acc(preds, masks).item() * imgs.size(0)
                vn += imgs.size(0)

        v_loss    /= vn
        v_dice     = dice_sum / vn
        v_pix_acc  = pix_sum  / vn
        scheduler.step()

        print(f"Epoch {epoch:03d} | Train {t_loss:.4f} | Val {v_loss:.4f} | "
              f"Dice {v_dice:.4f} | PixAcc {v_pix_acc:.4f}")

        wandb.log({
            "epoch":              epoch,
            "train/loss":         t_loss,
            "val/loss":           v_loss,
            "val/dice":           v_dice,
            "val/pixel_accuracy": v_pix_acc,
            "lr":                 scheduler.get_last_lr()[0],
        })

        if v_dice > best_dice:
            best_dice = v_dice
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), "checkpoints/unet.pth")
            print(f"  ✓ Saved (dice={v_dice:.4f}, pix_acc={v_pix_acc:.4f})")

    wandb.finish()
    print(f"\nBest val Dice: {best_dice:.4f}")


# ── Entry point ────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task",       type=str,   default="classification",
                   choices=["classification", "localization", "segmentation"])
    p.add_argument("--data_root",  type=str,   required=True)
    p.add_argument("--epochs",     type=int,   default=30)
    p.add_argument("--batch_size", type=int,   default=16)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--dropout_p",  type=float, default=0.5)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.task == "classification":
        train_classifier(args)
    elif args.task == "localization":
        train_localizer(args)
    elif args.task == "segmentation":
        train_segmentation(args)