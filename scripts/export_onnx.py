#!/usr/bin/env python3
"""Export the trained PyTorch checkpoints to ONNX.

Run this LOCALLY (only needs: pip install torch timm). It will download each
best.pt from the GitHub release if it is not already present, rebuild the
timm model, and write:

  runs/bikes_ensemble/<key>/best.onnx   <- upload these to your GitHub release
  api/onnx_meta.json                    <- commit this to the repo

The deployed API then serves with onnxruntime only (no torch/timm), which
fits comfortably in a 512MB instance.
"""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import torch
import timm

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "runs" / "bikes_ensemble"
META_PATH = ROOT / "api" / "onnx_meta.json"
# Upload-ready ONNX files land here with their flat release names so they can
# be dragged straight into the GitHub release without colliding.
UPLOAD_DIR = ROOT / "onnx_upload"

MODEL_KEYS = ["m1_convnext", "m2_effnetb0", "m3_vit_small"]
OPSET = 17

# Where to grab the original .pt checkpoints if they're not on disk locally.
RELEASE_BASE = "https://github.com/JuliJulia3/bmw_2/releases/download/weights-v2"
PT_URLS = {
    "m1_convnext": f"{RELEASE_BASE}/m1_convnext_best.pt",
    "m2_effnetb0": f"{RELEASE_BASE}/m2_effnetb0_best.pt",
    "m3_vit_small": f"{RELEASE_BASE}/m3_vit_small_best.pt",
}


def ensure_pt(model_key: str) -> Path:
    pt_path = RUNS_DIR / model_key / "best.pt"
    if pt_path.exists() and pt_path.stat().st_size > 0:
        return pt_path
    pt_path.parent.mkdir(parents=True, exist_ok=True)
    url = PT_URLS[model_key]
    print(f"Downloading {url} -> {pt_path}")
    with urllib.request.urlopen(url) as r, open(pt_path, "wb") as f:
        f.write(r.read())
    return pt_path


def export_one(model_key: str) -> dict:
    ckpt_path = ensure_pt(model_key)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    arch = ckpt["arch"]
    class_names = list(ckpt["class_names"])
    img_size = int(ckpt["img_size"])

    model = timm.create_model(arch, pretrained=False, num_classes=len(class_names))
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    out_path = UPLOAD_DIR / f"{model_key}_best.onnx"
    dummy = torch.randn(1, 3, img_size, img_size)

    # Dynamic batch axis: multi-crop TTA feeds k images as one batch.
    # dynamo=False uses the legacy TorchScript exporter, which embeds all
    # weights inline as a single self-contained .onnx file. (The dynamo
    # exporter splits weights into an external .onnx.data file, which would
    # leave these uploads weightless.)
    torch.onnx.export(
        model,
        dummy,
        out_path.as_posix(),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=OPSET,
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"Exported {model_key}: {out_path} (arch={arch}, img_size={img_size})")

    return {"arch": arch, "class_names": class_names, "img_size": img_size}


def main():
    meta = {mk: export_one(mk) for mk in MODEL_KEYS}
    META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote metadata: {META_PATH}")
    print(f"\nUpload-ready ONNX files are in: {UPLOAD_DIR}")
    print("Next: drag all 3 files from that folder into the weights-v2 GitHub")
    print("release, then commit api/onnx_meta.json and push.")


if __name__ == "__main__":
    main()
