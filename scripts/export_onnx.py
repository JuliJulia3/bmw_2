#!/usr/bin/env python3
"""Export the trained PyTorch checkpoints to ONNX.

Run this LOCALLY (where torch + timm are installed). It reads each
runs/bikes_ensemble/<key>/best.pt, rebuilds the timm model, and writes:

  runs/bikes_ensemble/<key>/best.onnx   <- upload these to your GitHub release
  api/onnx_meta.json                    <- commit this to the repo

The deployed API then serves with onnxruntime only (no torch/timm), which
fits comfortably in a 512MB instance.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import timm

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "runs" / "bikes_ensemble"
META_PATH = ROOT / "api" / "onnx_meta.json"

MODEL_KEYS = ["m1_convnext", "m2_effnetb0", "m3_vit_small"]
OPSET = 17


def export_one(model_key: str) -> dict:
    ckpt_path = RUNS_DIR / model_key / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    arch = ckpt["arch"]
    class_names = list(ckpt["class_names"])
    img_size = int(ckpt["img_size"])

    model = timm.create_model(arch, pretrained=False, num_classes=len(class_names))
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()

    out_path = RUNS_DIR / model_key / "best.onnx"
    dummy = torch.randn(1, 3, img_size, img_size)

    # Dynamic batch axis: multi-crop TTA feeds k images as one batch.
    torch.onnx.export(
        model,
        dummy,
        out_path.as_posix(),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=OPSET,
        do_constant_folding=True,
    )
    print(f"Exported {model_key}: {out_path} (arch={arch}, img_size={img_size})")

    return {"arch": arch, "class_names": class_names, "img_size": img_size}


def main():
    meta = {mk: export_one(mk) for mk in MODEL_KEYS}
    META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote metadata: {META_PATH}")


if __name__ == "__main__":
    main()
