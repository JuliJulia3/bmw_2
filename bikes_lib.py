from __future__ import annotations
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_device_cpu():
    return torch.device("cpu")

def list_images(folder: Path) -> List[Path]:
    exts = {".jpg",".jpeg",".png",".webp",".bmp",".tif",".tiff"}
    if not folder.exists():
        return []
    return [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in exts]

class ImgDataset(Dataset):
    def __init__(self, items: List[Tuple[Path,int]], tfm):
        self.items = items
        self.tfm = tfm

    def __len__(self): 
        return len(self.items)

    def __getitem__(self, idx):
        p, y = self.items[idx]
        img = Image.open(p).convert("RGB")
        x = self.tfm(img)
        return x, y, str(p)

def build_train_tfm(img_size: int):
    return transforms.Compose([
        transforms.RandomResizedCrop(
            img_size, scale=(0.45, 1.0), ratio=(0.75, 1.33),
            interpolation=transforms.InterpolationMode.BICUBIC
        ),
        transforms.RandomHorizontalFlip(p=0.5),

        transforms.RandomApply([
            transforms.RandomPerspective(distortion_scale=0.35, p=1.0),
        ], p=0.30),

        transforms.RandomApply([
            transforms.RandomAffine(
                degrees=14, translate=(0.06, 0.06), scale=(0.85, 1.15), shear=7
            )
        ], p=0.30),

        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.22, hue=0.05),
        transforms.RandomGrayscale(p=0.10),

        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),

        transforms.RandomErasing(p=0.20, scale=(0.02, 0.12), ratio=(0.3, 3.3), value="random"),
    ])

def build_val_tfm(img_size: int):
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.18), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

def multi_crop_tensors(img: Image.Image, img_size: int, k: int) -> torch.Tensor:
    resized = transforms.Resize(int(img_size * 1.28), interpolation=transforms.InterpolationMode.BICUBIC)(img)
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

    to_t = transforms.ToTensor()
    norm = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    stack = [norm(to_t(c)) for c in crops]
    return torch.stack(stack, dim=0)

@torch.no_grad()
def evaluate(model: nn.Module, loader, device: torch.device):
    model.eval()
    ce = nn.CrossEntropyLoss()
    total, correct = 0, 0
    loss_sum = 0.0
    all_y, all_p = [], []
    for x, y, _ in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = ce(logits, y)
        loss_sum += float(loss.item()) * x.size(0)
        pred = logits.argmax(dim=1)
        correct += int((pred == y).sum().item())
        total += x.size(0)
        all_y.extend(y.detach().cpu().tolist())
        all_p.extend(pred.detach().cpu().tolist())
    return loss_sum / max(1, total), correct / max(1, total), all_y, all_p

def save_ckpt(path: Path, *, model_key: str, class_names, img_size: int,
              arch: str, state_dict, extra):
    torch.save({
        "model_key": model_key,
        "arch": arch,
        "class_names": class_names,
        "img_size": img_size,
        "state_dict": state_dict,
        "extra": extra,
    }, path)
