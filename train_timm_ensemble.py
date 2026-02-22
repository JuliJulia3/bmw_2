#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import timm

from bikes_config import (
    CLEAN_DIR, RUNS_DIR, CLASSES, IMG_SIZE, BATCH, EPOCHS, LR, WEIGHT_DECAY, SEED,
    TEST_SIZE, VAL_SIZE_FROM_TRAIN
)
from bikes_lib import (
    seed_everything, get_device_cpu, list_images, ImgDataset,
    build_train_tfm, build_val_tfm, evaluate, save_ckpt
)

def build_model(arch: str, num_classes: int) -> nn.Module:
    return timm.create_model(arch, pretrained=True, num_classes=num_classes)

def train_one(arch: str, model_key: str):
    seed_everything(SEED)
    device = get_device_cpu()
    print("\n" + "="*90)
    print("Device:", device)
    print("Model:", model_key, "| Arch:", arch)

    run_dir = RUNS_DIR / model_key
    run_dir.mkdir(parents=True, exist_ok=True)
    best_path = run_dir / "best.pt"
    last_path = run_dir / "last.pt"

    # gather items
    items: List[Tuple[Path,int]] = []
    for i, c in enumerate(CLASSES):
        folder = CLEAN_DIR / c
        imgs = list_images(folder)
        if not imgs:
            raise RuntimeError(f"No images found in: {folder}")
        for p in imgs:
            items.append((p, i))

    y = np.array([yy for _, yy in items])
    print("Total images:", len(items))
    print("Per-class:", {c: int((y==i).sum()) for i,c in enumerate(CLASSES)})

    idx = np.arange(len(items))
    train_idx, test_idx = train_test_split(idx, test_size=TEST_SIZE, stratify=y, random_state=SEED)
    train_idx, val_idx = train_test_split(train_idx, test_size=VAL_SIZE_FROM_TRAIN, stratify=y[train_idx], random_state=SEED)

    train_items = [items[i] for i in train_idx]
    val_items   = [items[i] for i in val_idx]
    test_items  = [items[i] for i in test_idx]
    print(f"Split: train={len(train_items)} val={len(val_items)} test={len(test_items)}")

    # balanced sampler (now across 3 classes)
    train_y = [yy for _, yy in train_items]
    class_counts = np.bincount(train_y, minlength=len(CLASSES))
    weights = [1.0 / class_counts[yy] for yy in train_y]
    sampler = WeightedRandomSampler(weights, num_samples=len(train_items), replacement=True)

    train_ds = ImgDataset(train_items, build_train_tfm(IMG_SIZE))
    val_ds   = ImgDataset(val_items, build_val_tfm(IMG_SIZE))
    test_ds  = ImgDataset(test_items, build_val_tfm(IMG_SIZE))

    # CPU/macOS: num_workers=0 is usually best
    train_loader = DataLoader(train_ds, batch_size=BATCH, sampler=sampler, num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds, batch_size=BATCH, shuffle=False, num_workers=0)

    model = build_model(arch, num_classes=len(CLASSES)).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    ce = nn.CrossEntropyLoss(label_smoothing=0.05)

    best_val = -1.0
    for ep in range(1, EPOCHS+1):
        model.train()
        tot, correct, loss_sum = 0, 0, 0.0

        for x, yb, _ in train_loader:
            x = x.to(device)
            yb = yb.to(device)

            logits = model(x)
            loss = ce(logits, yb)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            pred = logits.argmax(dim=1)
            correct += int((pred == yb).sum().item())
            loss_sum += float(loss.item()) * x.size(0)
            tot += x.size(0)

        sched.step()

        train_loss = loss_sum / max(1, tot)
        train_acc = correct / max(1, tot)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, device)

        print(f"[{model_key}][{ep:02d}/{EPOCHS}] train loss={train_loss:.4f} acc={train_acc:.3f} | "
              f"val loss={val_loss:.4f} acc={val_acc:.3f}")

        save_ckpt(last_path, model_key=model_key, class_names=CLASSES, img_size=IMG_SIZE,
                  arch=arch, state_dict=model.state_dict(), extra={"epoch": ep})

        if val_acc > best_val:
            best_val = val_acc
            save_ckpt(best_path, model_key=model_key, class_names=CLASSES, img_size=IMG_SIZE,
                      arch=arch, state_dict=model.state_dict(),
                      extra={"epoch": ep, "best_val_acc": float(best_val)})

    print("\n=== Test (best) ===")
    ckpt = torch.load(best_path, map_location="cpu")
    model = build_model(ckpt["arch"], num_classes=len(ckpt["class_names"])).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()

    test_loss, test_acc, y_true, y_pred = evaluate(model, test_loader, device)
    print(f"Test loss={test_loss:.4f} acc={test_acc:.3f}")
    print("CM:\n", confusion_matrix(y_true, y_pred))
    print("\n", classification_report(y_true, y_pred, target_names=CLASSES, digits=3))
    print("Saved:", best_path)

def main():
    models = [
        ("convnext_tiny.fb_in22k_ft_in1k", "m1_convnext"),
        ("efficientnet_b0.ra_in1k", "m2_effnetb0"),
        ("vit_small_patch16_224.augreg_in21k_ft_in1k", "m3_vit_small"),
    ]
    for arch, key in models:
        train_one(arch, key)

if __name__ == "__main__":
    main()
