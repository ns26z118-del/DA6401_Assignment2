"""train.py — Training script for DA6401 Assignment 2.

Usage:
    python train.py --task classification --data_root /path/to/oxford-pet --epochs 30
    python train.py --task localization   --data_root /path/to/oxford-pet --epochs 20
    python train.py --task segmentation   --data_root /path/to/oxford-pet --epochs 20

Tracks all runs with Weights & Biases.
"""

import argparse
import os

import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader
from torchvision import transforms

from data.pets_dataset import OxfordIIITPetDataset
from models import VGG11Classifier, VGG11Localizer, VGG11UNet
from losses.iou_loss import IoULoss


# ── ImageNet normalisation stats (standard for VGG) ───────────────────────────
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]


def get_transforms(split: str):
    """Return augmentation + normalisation pipeline."""
    if split == "train":
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
        ])


# ── Classification training ────────────────────────────────────────────────────

def train_classifier(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    wandb.init(project="da6401-a2", name=f"classifier-drop{args.dropout_p}", config=vars(args))

    train_ds = OxfordIIITPetDataset(args.data_root, split="train", transform=get_transforms("train"))
    val_ds   = OxfordIIITPetDataset(args.data_root, split="val",   transform=get_transforms("val"))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = VGG11Classifier(num_classes=37, dropout_p=args.dropout_p).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        train_loss, correct, total = 0.0, 0, 0

        for batch in train_loader:
            imgs   = batch["image"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            logits = model(imgs)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * imgs.size(0)
            preds  = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += imgs.size(0)

        train_loss /= total
        train_acc   = correct / total

        # ── Validate ────────────────────────────────────────────────────────
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for batch in val_loader:
                imgs   = batch["image"].to(device)
                labels = batch["label"].to(device)
                logits = model(imgs)
                loss   = criterion(logits, labels)

                val_loss    += loss.item() * imgs.size(0)
                preds        = logits.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total   += imgs.size(0)

        val_loss /= val_total
        val_acc   = val_correct / val_total

        scheduler.step()

        print(f"Epoch {epoch:03d} | Train loss {train_loss:.4f} acc {train_acc:.4f} | "
              f"Val loss {val_loss:.4f} acc {val_acc:.4f}")

        wandb.log({
            "epoch": epoch,
            "train/loss": train_loss, "train/acc": train_acc,
            "val/loss":   val_loss,   "val/acc":   val_acc,
            "lr": scheduler.get_last_lr()[0],
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), "checkpoints/classifier.pth")
            print(f"  ✓ Saved best model (val_acc={val_acc:.4f})")

    wandb.finish()
    print(f"\nBest val accuracy: {best_val_acc:.4f}")


# ── Localization training ──────────────────────────────────────────────────────

def train_localizer(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wandb.init(project="da6401-a2", name="localizer", config=vars(args))

    train_ds = OxfordIIITPetDataset(args.data_root, split="train", transform=get_transforms("train"))
    val_ds   = OxfordIIITPetDataset(args.data_root, split="val",   transform=get_transforms("val"))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = VGG11Localizer(dropout_p=args.dropout_p).to(device)

    mse_loss = nn.MSELoss()
    iou_loss = IoULoss(reduction="mean")

    # Fine-tune entire network (encoder + head) — see localization.py for reasoning
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n = 0

        for batch in train_loader:
            imgs   = batch["image"].to(device)
            bboxes = batch["bbox"].to(device)

            optimizer.zero_grad()
            pred = model(imgs)

            loss = mse_loss(pred, bboxes) + iou_loss(pred, bboxes)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            n += imgs.size(0)

        train_loss = total_loss / n

        model.eval()
        val_loss_sum = 0.0
        val_n = 0
        with torch.no_grad():
            for batch in val_loader:
                imgs   = batch["image"].to(device)
                bboxes = batch["bbox"].to(device)
                pred   = model(imgs)
                loss   = mse_loss(pred, bboxes) + iou_loss(pred, bboxes)
                val_loss_sum += loss.item() * imgs.size(0)
                val_n += imgs.size(0)

        val_loss = val_loss_sum / val_n
        scheduler.step()

        print(f"Epoch {epoch:03d} | Train loss {train_loss:.4f} | Val loss {val_loss:.4f}")
        wandb.log({"epoch": epoch, "train/loss": train_loss, "val/loss": val_loss})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), "checkpoints/localizer.pth")

    wandb.finish()


# ── Entry point ────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task",      type=str, default="classification",
                   choices=["classification", "localization", "segmentation"])
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--epochs",    type=int, default=30)
    p.add_argument("--batch_size",type=int, default=32)
    p.add_argument("--lr",        type=float, default=1e-3)
    p.add_argument("--dropout_p", type=float, default=0.5)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.task == "classification":
        train_classifier(args)
    elif args.task == "localization":
        train_localizer(args)
    else:
        print("Segmentation training — implement after Task 3 (segmentation.py)")