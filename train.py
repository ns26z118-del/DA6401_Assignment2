"""train.py — Training script for DA6401 Assignment 2.

Usage:
    python train.py --task classification --data_root ./oxford-pet --epochs 50
    python train.py --task localization   --data_root ./oxford-pet --epochs 20
    python train.py --task segmentation   --data_root ./oxford-pet --epochs 20
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


# ── ImageNet normalisation stats ───────────────────────────────────────────────
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]


def get_transforms(split: str):
    if split == "train":
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
        ])


# ── Classification ─────────────────────────────────────────────────────────────

def train_classifier(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training classifier on: {device}")

    wandb.init(project="da6401-a2", name=f"classifier-drop{args.dropout_p}", config=vars(args))

    train_ds = OxfordIIITPetDataset(args.data_root, split="train", transform=get_transforms("train"))
    val_ds   = OxfordIIITPetDataset(args.data_root, split="val",   transform=get_transforms("val"))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model     = VGG11Classifier(num_classes=37, dropout_p=args.dropout_p).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
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
            correct    += (logits.argmax(1) == labels).sum().item()
            total      += imgs.size(0)

        train_loss /= total
        train_acc   = correct / total

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for batch in val_loader:
                imgs   = batch["image"].to(device)
                labels = batch["label"].to(device)
                logits = model(imgs)
                loss   = criterion(logits, labels)
                val_loss    += loss.item() * imgs.size(0)
                val_correct += (logits.argmax(1) == labels).sum().item()
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


# ── Localization ───────────────────────────────────────────────────────────────

def train_localizer(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training localizer on: {device}")

    wandb.init(project="da6401-a2", name="localizer", config=vars(args))

    train_ds = OxfordIIITPetDataset(args.data_root, split="train", transform=get_transforms("train"))
    val_ds   = OxfordIIITPetDataset(args.data_root, split="val",   transform=get_transforms("val"))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = VGG11Localizer(dropout_p=args.dropout_p).to(device)

    if os.path.exists("checkpoints/classifier.pth"):
        model.load_encoder_weights("checkpoints/classifier.pth")
    else:
        print("WARNING: classifier.pth not found — training encoder from scratch")

    mse_loss = nn.MSELoss()
    iou_loss = IoULoss(reduction="mean")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss, n = 0.0, 0

        for batch in train_loader:
            imgs   = batch["image"].to(device)
            bboxes = batch["bbox"].to(device)

            optimizer.zero_grad()
            pred = model(imgs)
            loss = mse_loss(pred, bboxes) + iou_loss(pred, bboxes)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * imgs.size(0)
            n += imgs.size(0)

        train_loss /= n

        model.eval()
        val_loss_sum, val_n = 0.0, 0

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
        wandb.log({"epoch": epoch, "train/loss": train_loss, "val/loss": val_loss,
                   "lr": scheduler.get_last_lr()[0]})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), "checkpoints/localizer.pth")
            print(f"  ✓ Saved best model (val_loss={val_loss:.4f})")

    wandb.finish()
    print(f"\nBest val loss: {best_val_loss:.4f}")


# ── Segmentation ───────────────────────────────────────────────────────────────

def train_segmentation(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training segmentation on: {device}")

    wandb.init(project="da6401-a2", name="segmentation", config=vars(args))

    train_ds = OxfordIIITPetDataset(args.data_root, split="train", transform=get_transforms("train"))
    val_ds   = OxfordIIITPetDataset(args.data_root, split="val",   transform=get_transforms("val"))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = VGG11UNet(num_classes=3, dropout_p=args.dropout_p).to(device)

    if os.path.exists("checkpoints/classifier.pth"):
        model.load_encoder_weights("checkpoints/classifier.pth")
    else:
        print("WARNING: classifier.pth not found — training encoder from scratch")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_dice = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss, n = 0.0, 0

        for batch in train_loader:
            imgs  = batch["image"].to(device)
            masks = batch["mask"].to(device)

            optimizer.zero_grad()
            logits = model(imgs)
            loss   = criterion(logits, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * imgs.size(0)
            n += imgs.size(0)

        train_loss /= n

        model.eval()
        val_loss_sum, dice_sum, val_n = 0.0, 0.0, 0

        with torch.no_grad():
            for batch in val_loader:
                imgs  = batch["image"].to(device)
                masks = batch["mask"].to(device)

                logits = model(imgs)
                loss   = criterion(logits, masks)
                val_loss_sum += loss.item() * imgs.size(0)

                preds = logits.argmax(dim=1)
                dice = 0.0
                for c in range(3):
                    pred_c = (preds == c).float()
                    true_c = (masks == c).float()
                    intersection = (pred_c * true_c).sum()
                    dice += (2 * intersection + 1e-6) / (pred_c.sum() + true_c.sum() + 1e-6)
                dice_sum += (dice / 3).item() * imgs.size(0)
                val_n += imgs.size(0)

        val_loss = val_loss_sum / val_n
        val_dice = dice_sum / val_n
        scheduler.step()

        print(f"Epoch {epoch:03d} | Train loss {train_loss:.4f} | "
              f"Val loss {val_loss:.4f} | Val Dice {val_dice:.4f}")

        wandb.log({
            "epoch": epoch,
            "train/loss": train_loss,
            "val/loss":   val_loss,
            "val/dice":   val_dice,
            "lr": scheduler.get_last_lr()[0],
        })

        if val_dice > best_dice:
            best_dice = val_dice
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), "checkpoints/unet.pth")
            print(f"  ✓ Saved best model (dice={val_dice:.4f})")

    wandb.finish()
    print(f"\nBest val Dice: {best_dice:.4f}")


# ── Entry point ────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task",       type=str,   default="classification",
                   choices=["classification", "localization", "segmentation"])
    p.add_argument("--data_root",  type=str,   required=True)
    p.add_argument("--epochs",     type=int,   default=30)
    p.add_argument("--batch_size", type=int,   default=32)
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

# """train.py — Training script for DA6401 Assignment 2.

# Usage:
#     python train.py --task classification --data_root /path/to/oxford-pet --epochs 30
#     python train.py --task localization   --data_root /path/to/oxford-pet --epochs 20
#     python train.py --task segmentation   --data_root /path/to/oxford-pet --epochs 20

# Tracks all runs with Weights & Biases.
# """

# import argparse
# import os

# import torch
# import torch.nn as nn
# import wandb
# from torch.utils.data import DataLoader
# from torchvision import transforms

# from data.pets_dataset import OxfordIIITPetDataset
# from models import VGG11Classifier, VGG11Localizer, VGG11UNet
# from losses.iou_loss import IoULoss


# # ── ImageNet normalisation stats (standard for VGG) ───────────────────────────
# MEAN = [0.485, 0.456, 0.406]
# STD  = [0.229, 0.224, 0.225]


# def get_transforms(split: str):
#     """Return augmentation + normalisation pipeline."""
#     if split == "train":
#         return transforms.Compose([
#             transforms.RandomHorizontalFlip(),
#             transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=MEAN, std=STD),
#         ])
#     else:
#         return transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize(mean=MEAN, std=STD),
#         ])


# # ── Classification training ────────────────────────────────────────────────────

# def train_classifier(args):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Training on: {device}")

#     wandb.init(project="da6401-a2", name=f"classifier-drop{args.dropout_p}", config=vars(args))

#     train_ds = OxfordIIITPetDataset(args.data_root, split="train", transform=get_transforms("train"))
#     val_ds   = OxfordIIITPetDataset(args.data_root, split="val",   transform=get_transforms("val"))

#     train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=4, pin_memory=True)
#     val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

#     model = VGG11Classifier(num_classes=37, dropout_p=args.dropout_p).to(device)

#     criterion = nn.CrossEntropyLoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
#     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

#     best_val_acc = 0.0

#     for epoch in range(1, args.epochs + 1):
#         # ── Train ──────────────────────────────────────────────────────────
#         model.train()
#         train_loss, correct, total = 0.0, 0, 0

#         for batch in train_loader:
#             imgs   = batch["image"].to(device)
#             labels = batch["label"].to(device)

#             optimizer.zero_grad()
#             logits = model(imgs)
#             loss   = criterion(logits, labels)
#             loss.backward()
#             optimizer.step()

#             train_loss += loss.item() * imgs.size(0)
#             preds  = logits.argmax(dim=1)
#             correct += (preds == labels).sum().item()
#             total   += imgs.size(0)

#         train_loss /= total
#         train_acc   = correct / total

#         # ── Validate ────────────────────────────────────────────────────────
#         model.eval()
#         val_loss, val_correct, val_total = 0.0, 0, 0

#         with torch.no_grad():
#             for batch in val_loader:
#                 imgs   = batch["image"].to(device)
#                 labels = batch["label"].to(device)
#                 logits = model(imgs)
#                 loss   = criterion(logits, labels)

#                 val_loss    += loss.item() * imgs.size(0)
#                 preds        = logits.argmax(dim=1)
#                 val_correct += (preds == labels).sum().item()
#                 val_total   += imgs.size(0)

#         val_loss /= val_total
#         val_acc   = val_correct / val_total

#         scheduler.step()

#         print(f"Epoch {epoch:03d} | Train loss {train_loss:.4f} acc {train_acc:.4f} | "
#               f"Val loss {val_loss:.4f} acc {val_acc:.4f}")

#         wandb.log({
#             "epoch": epoch,
#             "train/loss": train_loss, "train/acc": train_acc,
#             "val/loss":   val_loss,   "val/acc":   val_acc,
#             "lr": scheduler.get_last_lr()[0],
#         })

#         if val_acc > best_val_acc:
#             best_val_acc = val_acc
#             os.makedirs("checkpoints", exist_ok=True)
#             torch.save(model.state_dict(), "checkpoints/classifier.pth")
#             print(f"  ✓ Saved best model (val_acc={val_acc:.4f})")

#     wandb.finish()
#     print(f"\nBest val accuracy: {best_val_acc:.4f}")


# # ── Localization training ──────────────────────────────────────────────────────

# def train_localizer(args):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     wandb.init(project="da6401-a2", name="localizer", config=vars(args))

#     train_ds = OxfordIIITPetDataset(args.data_root, split="train", transform=get_transforms("train"))
#     val_ds   = OxfordIIITPetDataset(args.data_root, split="val",   transform=get_transforms("val"))

#     train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=4, pin_memory=True)
#     val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

#     model = VGG11Localizer(dropout_p=args.dropout_p).to(device)

#     mse_loss = nn.MSELoss()
#     iou_loss = IoULoss(reduction="mean")

#     # Fine-tune entire network (encoder + head) — see localization.py for reasoning
#     optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
#     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

#     best_val_loss = float("inf")

#     for epoch in range(1, args.epochs + 1):
#         model.train()
#         total_loss = 0.0
#         n = 0

#         for batch in train_loader:
#             imgs   = batch["image"].to(device)
#             bboxes = batch["bbox"].to(device)

#             optimizer.zero_grad()
#             pred = model(imgs)

#             loss = mse_loss(pred, bboxes) + iou_loss(pred, bboxes)
#             loss.backward()
#             optimizer.step()

#             total_loss += loss.item() * imgs.size(0)
#             n += imgs.size(0)

#         train_loss = total_loss / n

#         model.eval()
#         val_loss_sum = 0.0
#         val_n = 0
#         with torch.no_grad():
#             for batch in val_loader:
#                 imgs   = batch["image"].to(device)
#                 bboxes = batch["bbox"].to(device)
#                 pred   = model(imgs)
#                 loss   = mse_loss(pred, bboxes) + iou_loss(pred, bboxes)
#                 val_loss_sum += loss.item() * imgs.size(0)
#                 val_n += imgs.size(0)

#         val_loss = val_loss_sum / val_n
#         scheduler.step()

#         print(f"Epoch {epoch:03d} | Train loss {train_loss:.4f} | Val loss {val_loss:.4f}")
#         wandb.log({"epoch": epoch, "train/loss": train_loss, "val/loss": val_loss})

#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
#             os.makedirs("checkpoints", exist_ok=True)
#             torch.save(model.state_dict(), "checkpoints/localizer.pth")

#     wandb.finish()


# # ── Entry point ────────────────────────────────────────────────────────────────

# def parse_args():
#     p = argparse.ArgumentParser()
#     p.add_argument("--task",      type=str, default="classification",
#                    choices=["classification", "localization", "segmentation"])
#     p.add_argument("--data_root", type=str, required=True)
#     p.add_argument("--epochs",    type=int, default=30)
#     p.add_argument("--batch_size",type=int, default=32)
#     p.add_argument("--lr",        type=float, default=1e-3)
#     p.add_argument("--dropout_p", type=float, default=0.5)
#     return p.parse_args()


# if __name__ == "__main__":
#     args = parse_args()
#     if args.task == "classification":
#         train_classifier(args)
#     elif args.task == "localization":
#         train_localizer(args)
#     else:
#         print("Segmentation training — implement after Task 3 (segmentation.py)")