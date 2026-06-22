# predictor.py
from __future__ import annotations

import gc
import io
from dataclasses import dataclass
from typing import Dict, Any, List

import torch
from PIL import Image

# Keep memory low on small instances (e.g. Render 512Mi): a single thread
# avoids per-thread allocator arenas that inflate RSS.
torch.set_num_threads(1)

from bikes_config import CLASSES, MIN_CONF_FINAL, MIN_MARGIN_FINAL
from bikes_lib import get_device_cpu

# IMPORTANT: use the same loader + decision code as your ensemble script
from predict_ensemble import load_best, predict_with_model, manager_decision


@dataclass
class Prediction:
    status: str          # OK | RETAKE | UNKNOWN
    label: str           # r1300gs | r1300gs_adventure | UNKNOWN
    confidence: float
    margin: float
    reason: str
    per_model: Dict[str, Any]


class EnsemblePredictor:
    def __init__(self):
        self.device = get_device_cpu()
        self.class_names: List[str] = list(CLASSES)

        # must match your training run folders
        self.model_keys = ["m1_convnext", "m2_effnetb0", "m3_vit_small"]

    @torch.no_grad()
    def predict_pil(self, img: Image.Image) -> Prediction:
        per_model: Dict[str, Any] = {}

        # Load one model at a time, run it, then free it before loading the
        # next. Peak memory stays at ~1 model instead of 3, which keeps the
        # full ensemble within small instances (e.g. Render 512Mi). The cost
        # is re-reading each checkpoint from disk on every request.
        for mk in self.model_keys:
            model, class_names, img_size, _ckpt_path = load_best(mk, self.device)

            # safety: ensure class ordering matches exactly
            if list(class_names) != list(self.class_names):
                raise RuntimeError(
                    f"Class mismatch for {mk}: ckpt={class_names} vs config={self.class_names}"
                )

            try:
                pred, conf, margin, probs = predict_with_model(
                    model=model,
                    class_names=class_names,
                    img=img,
                    img_size=img_size,
                    device=self.device,
                )
            finally:
                # release the model before loading the next one
                del model
                gc.collect()

            # CRITICAL: keep probs so manager_decision fusion + thresholds behave the same
            per_model[mk] = {
                "pred": str(pred),
                "conf": float(conf),
                "margin": float(margin),
                "probs": [float(x) for x in probs],
            }

        final_label, final_conf, final_margin, reason = manager_decision(per_model, self.class_names)

        # Same UX rule you had
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
    its port before paying the (heavy) model-loading cost."""
    global _predictor
    if _predictor is None:
        _predictor = EnsemblePredictor()
    return _predictor