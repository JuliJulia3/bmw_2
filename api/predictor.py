# predictor.py
from __future__ import annotations

import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import onnxruntime as ort
from PIL import Image

from bikes_config import (
    CLASSES,
    MIN_CONF_FINAL,
    MIN_MARGIN_FINAL,
    MIN_CONF_PER_MODEL,
    RUNS_DIR,
    USE_MULTI_CROP,
    NUM_CROPS,
)

ROOT = Path(__file__).resolve().parent.parent
META_PATH = ROOT / "api" / "onnx_meta.json"

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Decision labels (mirror predict_ensemble.py — kept torch-free here so the
# serving image does not need torch).
OTHER_LABEL_NAME = "other"
UNKNOWN_NAME = "UNKNOWN"


@dataclass
class Prediction:
    status: str          # OK | RETAKE | UNKNOWN
    label: str           # r1300gs | r1300gs_adventure | UNKNOWN
    confidence: float
    margin: float
    reason: str
    per_model: Dict[str, Any]


# ----------------------------- preprocessing -----------------------------
# These mirror bikes_lib.build_val_tfm / multi_crop_tensors using PIL + numpy
# (no torchvision), so ONNX inference matches the original torch pipeline.

def _resize_shorter(img: Image.Image, target_short: int) -> Image.Image:
    """Resize so the shorter edge == target_short, preserving aspect ratio.
    Matches torchvision Resize(int) with BICUBIC."""
    w, h = img.size
    if w <= h:
        new_w = target_short
        new_h = int(round(target_short * h / w))
    else:
        new_h = target_short
        new_w = int(round(target_short * w / h))
    return img.resize((new_w, new_h), Image.BICUBIC)


def _center_crop(img: Image.Image, size: int) -> Image.Image:
    w, h = img.size
    left = (w - size) // 2
    top = (h - size) // 2
    return img.crop((left, top, left + size, top + size))


def _to_chw_normalized(img: Image.Image) -> np.ndarray:
    """PIL RGB image -> normalized CHW float32 array (ToTensor + Normalize)."""
    arr = np.asarray(img, dtype=np.float32) / 255.0     # HWC, [0,1]
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD           # broadcast over channels
    return np.transpose(arr, (2, 0, 1))                  # CHW


def _val_batch(img: Image.Image, img_size: int) -> np.ndarray:
    resized = _resize_shorter(img, int(img_size * 1.18))
    cropped = _center_crop(resized, img_size)
    x = _to_chw_normalized(cropped)
    return x[None, ...]                                   # [1,3,H,W]


def _multi_crop_batch(img: Image.Image, img_size: int, k: int) -> np.ndarray:
    resized = _resize_shorter(img, int(img_size * 1.28))
    W, H = resized.size
    s = img_size

    def crop(x1, y1):
        return resized.crop((x1, y1, x1 + s, y1 + s))

    cx1 = max(0, (W - s) // 2)
    cy1 = max(0, (H - s) // 2)
    crops = [
        crop(cx1, cy1),                      # center
        crop(0, 0),                          # top-left
        crop(max(0, W - s), 0),              # top-right
        crop(0, max(0, H - s)),              # bottom-left
        crop(max(0, W - s), max(0, H - s)),  # bottom-right
    ][:k]
    return np.stack([_to_chw_normalized(c) for c in crops], axis=0)  # [k,3,H,W]


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - np.max(logits)
    e = np.exp(z)
    return e / np.sum(e)


# ----------------------------- decision logic -----------------------------
# Torch-free copy of predict_ensemble.manager_decision.

def manager_decision(per_model: Dict[str, Dict], class_names: List[str]):
    votes = [(k, r["pred"], float(r["conf"]), float(r["margin"]))
             for k, r in per_model.items() if float(r["conf"]) >= MIN_CONF_PER_MODEL]

    if len(votes) >= 2:
        labels = [p for _, p, _, _ in votes]
        if labels.count(OTHER_LABEL_NAME) >= 2:
            confs = [c for _, p, c, _ in votes if p == OTHER_LABEL_NAME]
            avg_conf = sum(confs) / len(confs)
            return UNKNOWN_NAME, avg_conf, 0.0, "other_consensus"

    fused = [0.0] * len(class_names)
    wsum = 0.0
    for _, r in per_model.items():
        w = max(0.0, float(r["conf"]))
        for i in range(len(class_names)):
            fused[i] += w * float(r["probs"][i])
        wsum += w
    fused = [v / wsum for v in fused] if wsum > 1e-9 else [1 / len(class_names)] * len(class_names)

    top_idx = max(range(len(fused)), key=lambda i: fused[i])
    top_label = class_names[top_idx]
    top_conf = float(fused[top_idx])
    sorted_vals = sorted(fused, reverse=True)
    fused_margin = float(sorted_vals[0] - sorted_vals[1]) if len(sorted_vals) > 1 else top_conf

    if top_label == OTHER_LABEL_NAME:
        return UNKNOWN_NAME, top_conf, fused_margin, "other_fused"

    if top_conf < MIN_CONF_FINAL or fused_margin < MIN_MARGIN_FINAL:
        return UNKNOWN_NAME, top_conf, fused_margin, "ambiguous_gate"

    return top_label, top_conf, fused_margin, "fused_ok"


# ----------------------------- predictor -----------------------------

class EnsemblePredictor:
    def __init__(self):
        self.class_names: List[str] = list(CLASSES)
        self.model_keys = ["m1_convnext", "m2_effnetb0", "m3_vit_small"]

        self.meta: Dict[str, Dict] = json.loads(META_PATH.read_text(encoding="utf-8"))

        # Keep all sessions loaded, but minimise resident memory for a 512MB
        # instance: single-threaded, and no CPU arena (the arena reserves and
        # holds large blocks that push RSS over the limit).
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        so.enable_cpu_mem_arena = False

        self.sessions: Dict[str, ort.InferenceSession] = {}
        for mk in self.model_keys:
            onnx_path = RUNS_DIR / mk / "best.onnx"
            self.sessions[mk] = ort.InferenceSession(
                onnx_path.as_posix(),
                sess_options=so,
                providers=["CPUExecutionProvider"],
            )

            ckpt_classes = list(self.meta[mk]["class_names"])
            if ckpt_classes != self.class_names:
                raise RuntimeError(
                    f"Class mismatch for {mk}: onnx={ckpt_classes} vs config={self.class_names}"
                )

    def _predict_one(self, mk: str, img: Image.Image):
        img_size = int(self.meta[mk]["img_size"])
        if USE_MULTI_CROP:
            x = _multi_crop_batch(img, img_size, NUM_CROPS)
        else:
            x = _val_batch(img, img_size)

        sess = self.sessions[mk]
        input_name = sess.get_inputs()[0].name

        # Run crops ONE AT A TIME (batch=1) and average the logits. This is
        # numerically identical to a single batched forward but keeps peak
        # activation memory ~Nx smaller, which matters on a 512MB instance.
        logits_sum = None
        for i in range(x.shape[0]):
            xi = x[i:i + 1].astype(np.float32)            # [1,3,H,W]
            li = sess.run(None, {input_name: xi})[0][0]   # [C]
            logits_sum = li if logits_sum is None else logits_sum + li
        logits = logits_sum / x.shape[0]
        probs = _softmax(logits)

        idx = int(np.argmax(probs))
        conf = float(probs[idx])
        pred = self.class_names[idx]

        top2 = np.sort(probs)[::-1][:2]
        margin = float(top2[0] - top2[1]) if len(top2) >= 2 else conf
        return pred, conf, margin, probs.tolist()

    def predict_pil(self, img: Image.Image) -> Prediction:
        per_model: Dict[str, Any] = {}
        for mk in self.model_keys:
            pred, conf, margin, probs = self._predict_one(mk, img)
            per_model[mk] = {
                "pred": str(pred),
                "conf": float(conf),
                "margin": float(margin),
                "probs": [float(x) for x in probs],
            }

        final_label, final_conf, final_margin, reason = manager_decision(per_model, self.class_names)

        if str(final_label) == "UNKNOWN":
            if (float(final_conf) < float(MIN_CONF_FINAL)) or (float(final_margin) < float(MIN_MARGIN_FINAL)):
                status = "RETAKE"
            else:
                status = "UNKNOWN"
        else:
            status = "OK"

        return Prediction(
            status=status,
            label=str(final_label),
            confidence=float(final_conf),
            margin=float(final_margin),
            reason=str(reason),
            per_model={k: {kk: vv for kk, vv in v.items() if kk != "probs"} for k, v in per_model.items()},
        )

    def predict_bytes(self, image_bytes: bytes) -> Prediction:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return self.predict_pil(img)


_predictor: EnsemblePredictor | None = None


def get_predictor() -> EnsemblePredictor:
    """Lazily build the predictor on first use so the web process can bind
    its port before paying the model-loading cost."""
    global _predictor
    if _predictor is None:
        _predictor = EnsemblePredictor()
    return _predictor
