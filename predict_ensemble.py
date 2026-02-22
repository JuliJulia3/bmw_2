#!/usr/bin/env python3
from __future__ import annotations

import csv
import shutil
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
from PIL import Image
import timm

from bikes_config import (
    RUNS_DIR, PREDICT_DIR, OUT_CSV, SORT_OUT, SORT_DIR,
    MIN_CONF_FINAL, MIN_MARGIN_FINAL, MIN_CONF_PER_MODEL,
    USE_MULTI_CROP, NUM_CROPS, CLASSES
)
from bikes_lib import get_device_cpu, list_images, build_val_tfm, multi_crop_tensors


# If model predicts "other", we output UNKNOWN (your desired behavior)
OTHER_LABEL_NAME = "other"
UNKNOWN_NAME = "UNKNOWN"


def build_model(arch: str, num_classes: int):
    return timm.create_model(arch, pretrained=False, num_classes=num_classes)

def load_best(model_key: str, device: torch.device):
    ckpt_path = RUNS_DIR / model_key / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path} (train first)")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = build_model(ckpt["arch"], num_classes=len(ckpt["class_names"]))
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.to(device).eval()
    return model, ckpt["class_names"], ckpt["img_size"], ckpt_path


def ensure_out_dirs():
    if not SORT_OUT:
        return
    (SORT_DIR / UNKNOWN_NAME).mkdir(parents=True, exist_ok=True)
    for c in CLASSES:
        # We'll still create other/ in case you want it later, but we’ll map it to UNKNOWN in output
        (SORT_DIR / c).mkdir(parents=True, exist_ok=True)

def sort_copy(img_path: Path, label: str):
    if not SORT_OUT:
        return
    dst = SORT_DIR / label / img_path.name
    if dst.exists():
        stem, suf = dst.stem, dst.suffix
        i = 2
        while (SORT_DIR / label / f"{stem}_{i}{suf}").exists():
            i += 1
        dst = SORT_DIR / label / f"{stem}_{i}{suf}"
    shutil.copy2(img_path, dst)


@torch.no_grad()
def predict_with_model(model, class_names: List[str], img: Image.Image, img_size: int, device: torch.device):
    if USE_MULTI_CROP:
        x = multi_crop_tensors(img, img_size=img_size, k=NUM_CROPS).to(device)
        logits_k = model(x)
        logits = logits_k.mean(dim=0)
        probs = F.softmax(logits, dim=0)
    else:
        tfm = build_val_tfm(img_size)
        x = tfm(img).unsqueeze(0).to(device)
        logits = model(x)[0]
        probs = F.softmax(logits, dim=0)

    conf, idx = torch.max(probs, dim=0)
    conf = float(conf.item())
    pred = class_names[int(idx.item())]

    top2 = torch.topk(probs, k=min(2, len(class_names))).values
    margin = float((top2[0] - top2[1]).item()) if len(top2) >= 2 else conf

    return pred, conf, margin, probs.detach().cpu().tolist()


def manager_decision(per_model: Dict[str, Dict], class_names: List[str]):
    # vote among confident models
    votes = [(k, r["pred"], float(r["conf"]), float(r["margin"]))
             for k, r in per_model.items() if float(r["conf"]) >= MIN_CONF_PER_MODEL]

    # if 2+ models say "other" => UNKNOWN
    if len(votes) >= 2:
        labels = [p for _, p, _, _ in votes]
        if labels.count(OTHER_LABEL_NAME) >= 2:
            confs = [c for _, p, c, _ in votes if p == OTHER_LABEL_NAME]
            avg_conf = sum(confs)/len(confs)
            return UNKNOWN_NAME, avg_conf, 0.0, "other_consensus"

    # fused probabilities weighted by confidence
    fused = [0.0] * len(class_names)
    wsum = 0.0
    for _, r in per_model.items():
        w = max(0.0, float(r["conf"]))
        for i in range(len(class_names)):
            fused[i] += w * float(r["probs"][i])
        wsum += w
    fused = [v / wsum for v in fused] if wsum > 1e-9 else [1/len(class_names)] * len(class_names)

    top_idx = max(range(len(fused)), key=lambda i: fused[i])
    top_label = class_names[top_idx]
    top_conf = float(fused[top_idx])
    sorted_vals = sorted(fused, reverse=True)
    fused_margin = float(sorted_vals[0] - sorted_vals[1]) if len(sorted_vals) > 1 else top_conf

    # if it predicts "other" with decent confidence => UNKNOWN
    if top_label == OTHER_LABEL_NAME:
        return UNKNOWN_NAME, top_conf, fused_margin, "other_fused"

    # otherwise apply normal gates
    if top_conf < MIN_CONF_FINAL or fused_margin < MIN_MARGIN_FINAL:
        return UNKNOWN_NAME, top_conf, fused_margin, "ambiguous_gate"

    return top_label, top_conf, fused_margin, "fused_ok"


def main():
    device = get_device_cpu()
    print("Device:", device)

    model_keys = ["m1_convnext", "m2_effnetb0", "m3_vit_small"]

    loaded = []
    for mk in model_keys:
        model, class_names, img_size, ckpt_path = load_best(mk, device)
        if class_names != CLASSES:
            raise RuntimeError(f"Class mismatch in {mk}: got {class_names}, expected {CLASSES}")
        loaded.append((mk, model, class_names, img_size))
        print(f"Loaded {mk}: {ckpt_path}")

    ensure_out_dirs()

    imgs = list_images(PREDICT_DIR)
    print(f"Found {len(imgs)} images in {PREDICT_DIR}/")
    if not imgs:
        return

    rows = []
    unknown = 0

    for p in sorted(imgs):
        img = Image.open(p).convert("RGB")
        per_model = {}

        for mk, model, class_names, img_size in loaded:
            pred, conf, margin, probs = predict_with_model(model, class_names, img, img_size, device)
            per_model[mk] = {"pred": pred, "conf": conf, "margin": margin, "probs": probs}

        final_label, final_conf, final_margin, decision_type = manager_decision(per_model, CLASSES)

        row = {
            "file": p.name,
            "final_label": final_label,
            "final_conf": f"{final_conf:.6f}",
            "final_margin": f"{final_margin:.6f}",
            "decision_type": decision_type,
        }

        for mk in model_keys:
            row[f"{mk}_pred"] = per_model[mk]["pred"]
            row[f"{mk}_conf"] = f'{per_model[mk]["conf"]:.6f}'
            row[f"{mk}_margin"] = f'{per_model[mk]["margin"]:.6f}'
            for i, c in enumerate(CLASSES):
                row[f"{mk}_prob_{c}"] = f'{float(per_model[mk]["probs"][i]):.6f}'

        rows.append(row)
        sort_copy(p, final_label)
        if final_label == UNKNOWN_NAME:
            unknown += 1

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"Saved CSV: {OUT_CSV}")
    print(f"Sorted into: {SORT_DIR}/ (UNKNOWN={unknown})")

if __name__ == "__main__":
    main()
