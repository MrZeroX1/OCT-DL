"""Consistent OCT loading and ImageNet-compatible preprocessing."""
from __future__ import annotations
from pathlib import Path
from typing import Tuple
import numpy as np
from PIL import Image, ImageOps
import torch
from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def load_oct_image(path: str | Path) -> Image.Image:
    """Load a genuine OCT image, applying EXIF orientation and RGB conversion."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"OCT image does not exist: {path}")
    try:
        with Image.open(path) as opened:
            opened.verify()
        with Image.open(path) as opened:
            return ImageOps.exif_transpose(opened).convert("RGB")
    except Exception as exc:
        raise ValueError(f"Unreadable OCT image {path}: {exc}") from exc

def image_statistics(image: Image.Image) -> dict:
    arr = np.asarray(image.convert("L"), dtype=np.float32)
    return {"width": image.width, "height": image.height, "channels": len(image.getbands()),
            "min": int(arr.min()), "max": int(arr.max()), "mean": float(arr.mean()), "std": float(arr.std()),
            "contrast_nonzero": bool(arr.max() > arr.min())}

def build_transform(image_size: int, training: bool = False) -> transforms.Compose:
    ops = [transforms.Resize((image_size, image_size), antialias=True)]
    if training:
        # Mild geometric changes; no vertical flip because OCT anatomy is directional.
        ops.extend([transforms.RandomRotation(5), transforms.RandomAffine(0, translate=(0.03, 0.03))])
    ops.extend([transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    return transforms.Compose(ops)

def tensor_from_path(path: str | Path, image_size: int) -> torch.Tensor:
    return build_transform(image_size, training=False)(load_oct_image(path))
