"""Model initialization, checkpoint loading, and preprocessing pipeline.

Ensures strict reproducibility matching unetpp-res34.ipynb and xai_gradcam_unetpp.ipynb.
Loads weights strictly from evaluation/unetpp_best.pth without re-training.
"""

from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import os
import random
import numpy as np
from PIL import Image
import cv2

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import segmentation_models_pytorch as smp

# Preprocessing Constants matching training notebook
IMAGE_SIZE = 448
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
DEFAULT_THRESHOLD = 0.50

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_CACHED_MODEL: Optional[nn.Module] = None
_CHECKPOINT_METADATA: Dict[str, Any] = {}


def get_model(checkpoint_path: Optional[str] = None) -> Tuple[nn.Module, Dict[str, Any]]:
    """Loads and caches the U-Net++ ResNet-34 model checkpoint.

    Args:
        checkpoint_path: Path to unetpp_best.pth. If None, resolves to evaluation/unetpp_best.pth.

    Returns:
        Tuple of (model, checkpoint_metadata_dict).
    """
    global _CACHED_MODEL, _CHECKPOINT_METADATA

    if _CACHED_MODEL is not None:
        return _CACHED_MODEL, _CHECKPOINT_METADATA

    if checkpoint_path is None:
        # Resolve relative to project root
        base_dir = Path(__file__).resolve().parent.parent.parent
        checkpoint_path = str(base_dir / "evaluation" / "unetpp_best.pth")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at '{checkpoint_path}'. Ensure evaluation/unetpp_best.pth exists.")

    print(f"[Model] Initializing U-Net++ (ResNet-34) on {DEVICE}...")
    model = smp.UnetPlusPlus(
        encoder_name="resnet34",
        encoder_weights=None,  # Loading trained checkpoint weights directly
        in_channels=3,
        classes=1,
    )

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    best_val_iou = checkpoint.get("best_val_iou", None)
    epoch = checkpoint.get("epoch", None)
    val_metrics = checkpoint.get("val_metrics", {})

    _CHECKPOINT_METADATA = {
        "architecture": "U-Net++",
        "encoder": "ResNet-34",
        "checkpoint_path": checkpoint_path,
        "device": str(DEVICE),
        "epoch": epoch,
        "best_val_iou": float(best_val_iou) if best_val_iou is not None else None,
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "val_metrics": {k: float(v) for k, v in val_metrics.items()} if isinstance(val_metrics, dict) else {},
    }
    _CACHED_MODEL = model
    print(f"[Model] Successfully loaded checkpoint from epoch {epoch} (Val IoU: {_CHECKPOINT_METADATA['best_val_iou']})")
    return _CACHED_MODEL, _CHECKPOINT_METADATA


def preprocess_image(
    pil_image: Image.Image,
    target_size: int = IMAGE_SIZE
) -> Tuple[torch.Tensor, np.ndarray]:
    """Applies the exact inference preprocessing pipeline from unetpp-res34.ipynb.

    1. RGB conversion
    2. Bilinear resize to 448x448
    3. Conversion to torch FloatTensor in [0.0, 1.0]
    4. ImageNet normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]

    Returns:
        Tuple of (normalized_tensor [1, 3, 448, 448], display_rgb_array [448, 448, 3] in [0, 255] uint8).
    """
    img_rgb = pil_image.convert("RGB")
    resized_img = TF.resize(img_rgb, [target_size, target_size], interpolation=TF.InterpolationMode.BILINEAR)

    # Tensor normalization
    tensor = TF.to_tensor(resized_img)
    norm_tensor = TF.normalize(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD)
    batch_tensor = norm_tensor.unsqueeze(0).to(DEVICE)

    display_array = np.array(resized_img, dtype=np.uint8)
    return batch_tensor, display_array


def simulate_training_augmentations(
    pil_image: Image.Image,
    target_size: int = IMAGE_SIZE,
    seed: Optional[int] = None
) -> np.ndarray:
    """Applies the four synchronized training augmentations defined in unetpp-res34.ipynb.

    Allows researcher to inspect the augmentation transformation visually:
    - Random Horizontal Flip (p=0.5)
    - Random Vertical Flip (p=0.5)
    - Random Resized Crop / Scale variation (pad 24, scale 0.85-1.0)
    - Photometric Jitter (brightness & contrast 0.8-1.2)
    """
    if seed is not None:
        random.seed(seed)

    img = pil_image.convert("RGB")
    img = TF.resize(img, [target_size, target_size])

    # 1. Flip augmentations
    if random.random() < 0.5:
        img = TF.hflip(img)
    if random.random() < 0.5:
        img = TF.vflip(img)

    # 2. Resized Crop
    pad = 24
    img = TF.pad(img, pad)
    i, j, h, w = transforms.RandomResizedCrop.get_params(
        img, scale=(0.85, 1.0), ratio=(1.0, 1.0)
    )
    img = TF.resized_crop(img, i, j, h, w, [target_size, target_size])

    # 3. Photometric Jitter
    b_factor = random.uniform(0.8, 1.2)
    c_factor = random.uniform(0.8, 1.2)
    img = TF.adjust_brightness(img, b_factor)
    img = TF.adjust_contrast(img, c_factor)

    return np.array(img, dtype=np.uint8)
