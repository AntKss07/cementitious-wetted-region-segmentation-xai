"""Inference and Explainable AI (XAI) pipeline orchestrator.

Implements the end-to-end scientific flow:
Input Image -> Preprocessing -> Segmentation -> Physics Waterline -> WAR Computation -> Grad-CAM++ -> Comparison Visualizations.
"""

from typing import Dict, Any, Optional, Tuple
import io
import base64
import time
import numpy as np
from PIL import Image
import cv2
import torch
import torchvision.transforms.functional as TF

from .model import (
    get_model,
    preprocess_image,
    simulate_training_augmentations,
    IMAGE_SIZE,
    DEFAULT_THRESHOLD,
    IMAGENET_MEAN,
    IMAGENET_STD,
    DEVICE,
)
from .physics import enforce_waterline_physics, compute_war, compute_metrics
from .gradcam import GradCAMPlusPlus, get_target_layer_by_name


def encode_image_base64(img_array: np.ndarray, format: str = "PNG") -> str:
    """Encodes a numpy RGB or RGBA uint8 array into a base64 data URI."""
    if img_array.dtype != np.uint8:
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)

    pil_img = Image.fromarray(img_array)
    buffer = io.BytesIO()
    pil_img.save(buffer, format=format)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    mime = "image/png" if format.upper() == "PNG" else "image/jpeg"
    return f"data:{mime};base64,{encoded}"


def create_mask_overlay(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    color: tuple = (16, 185, 129),  # Emerald RGB
    alpha: float = 0.50
) -> np.ndarray:
    """Blends a binary mask onto the RGB image."""
    overlay = image_rgb.copy()
    mask_indices = (mask > 0)
    for c in range(3):
        overlay[mask_indices, c] = (
            (1.0 - alpha) * overlay[mask_indices, c] + alpha * color[c]
        ).astype(np.uint8)
    return overlay


def create_heatmap_blend(
    image_rgb: np.ndarray,
    cam_norm: np.ndarray,
    colormap_name: str = "jet",
    alpha: float = 0.55
) -> Tuple[np.ndarray, np.ndarray]:
    """Applies scientific colormap to CAM and blends with the original RGB image.

    Returns:
        Tuple of (colored_heatmap_rgb [H, W, 3], blended_image_rgb [H, W, 3]).
    """
    cam_uint8 = np.uint8(255 * np.clip(cam_norm, 0.0, 1.0))

    cv_colormaps = {
        "jet": cv2.COLORMAP_JET,
        "plasma": cv2.COLORMAP_PLASMA,
        "inferno": cv2.COLORMAP_INFERNO,
        "viridis": cv2.COLORMAP_VIRIDIS,
    }
    cv_cmap = cv_colormaps.get(colormap_name.lower(), cv2.COLORMAP_JET)
    heatmap_bgr = cv2.applyColorMap(cam_uint8, cv_cmap)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    blended = (1.0 - alpha) * image_rgb.astype(np.float32) + alpha * heatmap_rgb.astype(np.float32)
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    return heatmap_rgb, blended


def create_error_map(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Creates a high-contrast publication error map:
    Green = True Positive (TP)
    Red   = False Positive (FP)
    Blue  = False Negative (FN)
    Dark  = True Negative (TN)
    """
    p = (pred > 0).astype(np.uint8)
    g = (gt > 0).astype(np.uint8)
    H, W = p.shape
    err_rgb = np.zeros((H, W, 3), dtype=np.uint8)

    err_rgb[(p == 1) & (g == 1)] = [16, 185, 129]  # Emerald Green TP
    err_rgb[(p == 1) & (g == 0)] = [239, 68, 68]   # Red FP
    err_rgb[(p == 0) & (g == 1)] = [59, 130, 246]  # Blue FN
    err_rgb[(p == 0) & (g == 0)] = [15, 23, 42]    # Dark Slate TN
    return err_rgb


class SorptivityPipeline:
    """Coordinates segmentation, physics constraints, WAR computation, and Grad-CAM++."""

    def __init__(self):
        self.model, self.metadata = get_model()

    def run_analysis(
        self,
        pil_image: Image.Image,
        gt_mask_pil: Optional[Image.Image] = None,
        target_layer_name: str = "decoder_x_0_4",
        pred_threshold: float = DEFAULT_THRESHOLD,
        apply_physics: bool = True,
        bottom_margin: int = 32,
        colormap: str = "jet",
        heatmap_alpha: float = 0.55,
        include_simulated_augmentation: bool = False
    ) -> Dict[str, Any]:
        """Executes the full pipeline on an input image.

        Returns comprehensive quantitative metrics and base64 encoded comparison imagery.
        """
        start_time = time.perf_counter()

        # 1. Preprocessing
        t_pre = time.perf_counter()
        img_tensor, display_rgb = preprocess_image(pil_image, target_size=IMAGE_SIZE)
        preproc_latency_ms = (time.perf_counter() - t_pre) * 1000

        # Optional simulated augmentation view
        augmented_preview = None
        if include_simulated_augmentation:
            aug_rgb = simulate_training_augmentations(pil_image, target_size=IMAGE_SIZE)
            augmented_preview = encode_image_base64(aug_rgb)

        # 2. Grad-CAM++ & Segmentation Logits
        t_infer = time.perf_counter()
        target_layer = get_target_layer_by_name(self.model, target_layer_name)
        cam_engine = GradCAMPlusPlus(self.model, target_layer)

        cam_norm, logits = cam_engine.generate(
            img_tensor,
            pred_threshold=pred_threshold
        )
        cam_engine.remove_hooks()
        inference_latency_ms = (time.perf_counter() - t_infer) * 1000

        # 3. Raw Mask Segmentation
        probs = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
        pred_raw_binary = (probs > pred_threshold).astype(np.uint8)

        # 4. Physics Domain Enforcement
        t_phys = time.perf_counter()
        if apply_physics:
            pred_phys_binary = enforce_waterline_physics(
                pred_raw_binary,
                bottom_margin=bottom_margin
            )
        else:
            pred_phys_binary = pred_raw_binary.copy()
        physics_latency_ms = (time.perf_counter() - t_phys) * 1000

        # 5. Downstream WAR Analysis
        total_pixels = int(display_rgb.shape[0] * display_rgb.shape[1])
        raw_wet_pixels = int(np.sum(pred_raw_binary > 0))
        phys_wet_pixels = int(np.sum(pred_phys_binary > 0))

        war_raw = compute_war(pred_raw_binary)
        war_phys = compute_war(pred_phys_binary)

        # Average absorption height (mean Y distance of wet front from specimen top)
        wet_rows = np.where(pred_phys_binary == 1)[0]
        mean_waterline_y = float(np.min(wet_rows)) if len(wet_rows) > 0 else float(IMAGE_SIZE)
        absorption_height_px = float(IMAGE_SIZE - mean_waterline_y)

        # 6. Heatmap generation
        heatmap_colored_rgb, heatmap_blended_rgb = create_heatmap_blend(
            display_rgb,
            cam_norm,
            colormap_name=colormap,
            alpha=heatmap_alpha
        )

        # Mask overlays
        mask_overlay_raw = create_mask_overlay(display_rgb, pred_raw_binary, color=(59, 130, 246), alpha=0.45)
        mask_overlay_phys = create_mask_overlay(display_rgb, pred_phys_binary, color=(16, 185, 129), alpha=0.45)

        # 7. Ground Truth Evaluation if provided
        gt_metrics = None
        error_map_b64 = None
        gt_mask_b64 = None
        war_gt = None

        if gt_mask_pil is not None:
            # Resize ground truth mask to target size with nearest neighbor
            gt_resized = TF.resize(gt_mask_pil.convert("L"), [IMAGE_SIZE, IMAGE_SIZE], interpolation=TF.InterpolationMode.NEAREST)
            gt_binary = (np.array(gt_resized) > 127).astype(np.uint8)
            war_gt = compute_war(gt_binary)

            metrics_raw = compute_metrics(pred_raw_binary, gt_binary)
            metrics_phys = compute_metrics(pred_phys_binary, gt_binary)

            gt_metrics = {
                "war_gt": float(war_gt),
                "war_abs_error_raw": float(abs(war_raw - war_gt)),
                "war_abs_error_phys": float(abs(war_phys - war_gt)),
                "raw": metrics_raw,
                "phys": metrics_phys,
            }

            err_rgb = create_error_map(pred_phys_binary, gt_binary)
            error_map_b64 = encode_image_base64(err_rgb)
            gt_mask_disp = np.stack([gt_binary * 255] * 3, axis=-1)
            gt_mask_b64 = encode_image_base64(gt_mask_disp)

        # Binary masks visual display (0 -> black, 1 -> white)
        raw_mask_disp = np.stack([pred_raw_binary * 255] * 3, axis=-1)
        phys_mask_disp = np.stack([pred_phys_binary * 255] * 3, axis=-1)

        total_latency_ms = (time.perf_counter() - start_time) * 1000

        return {
            "metadata": {
                "architecture": self.metadata.get("architecture", "U-Net++"),
                "encoder": self.metadata.get("encoder", "ResNet-34"),
                "device": str(DEVICE),
                "target_layer": target_layer_name,
                "image_size": [IMAGE_SIZE, IMAGE_SIZE],
                "confidence_threshold": float(pred_threshold),
                "applied_physics_constraints": bool(apply_physics),
                "bottom_margin_px": int(bottom_margin),
                "colormap": colormap,
                "total_latency_ms": round(total_latency_ms, 2),
                "breakdown_latency_ms": {
                    "preprocessing": round(preproc_latency_ms, 2),
                    "inference_and_xai": round(inference_latency_ms, 2),
                    "physics": round(physics_latency_ms, 2),
                }
            },
            "war_analysis": {
                "war_raw": float(war_raw),
                "war_raw_percent": round(war_raw * 100, 2),
                "war_physics": float(war_phys),
                "war_physics_percent": round(war_phys * 100, 2),
                "raw_wet_pixels": raw_wet_pixels,
                "physics_wet_pixels": phys_wet_pixels,
                "total_specimen_pixels": total_pixels,
                "absorption_height_px": round(absorption_height_px, 1),
                "filtered_floating_noise_pixels": max(0, raw_wet_pixels - phys_wet_pixels),
            },
            "ground_truth_evaluation": gt_metrics,
            "images": {
                "input_image": encode_image_base64(display_rgb),
                "raw_mask": encode_image_base64(raw_mask_disp),
                "physics_mask": encode_image_base64(phys_mask_disp),
                "raw_overlay": encode_image_base64(mask_overlay_raw),
                "physics_overlay": encode_image_base64(mask_overlay_phys),
                "gradcam_heatmap": encode_image_base64(heatmap_colored_rgb),
                "gradcam_blended": encode_image_base64(heatmap_blended_rgb),
                "error_map": error_map_b64,
                "ground_truth_mask": gt_mask_b64,
                "simulated_augmentation": augmented_preview,
            }
        }
