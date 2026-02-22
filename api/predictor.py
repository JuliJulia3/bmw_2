# predictor.py
from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple

import torch
from PIL import Image

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

        # store (model, class_names, img_size) per key
        self.loaded: Dict[str, Tuple[torch.nn.Module, List[str], int]] = {}
        self._load_models()

    def _load_models(self):
        for mk in self.model_keys:
            model, class_names, img_size, _ckpt_path = load_best(mk, self.device)

            # safety: ensure class ordering matches exactly
            if list(class_names) != list(self.class_names):
                raise RuntimeError(
                    f"Class mismatch for {mk}: ckpt={class_names} vs config={self.class_names}"
                )

            self.loaded[mk] = (model, class_names, img_size)

    @torch.no_grad()
    def predict_pil(self, img: Image.Image) -> Prediction:
        per_model: Dict[str, Any] = {}

        for mk, (model, class_names, img_size) in self.loaded.items():
            pred, conf, margin, probs = predict_with_model(
                model=model,
                class_names=class_names,
                img=img,
                img_size=img_size,
                device=self.device,
            )

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


predictor = EnsemblePredictor()