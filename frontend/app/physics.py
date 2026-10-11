"""Physics-informed post-processing and physical metric computation.

Based on domain physics of capillary water absorption in porous cementitious materials:
1. Bottom Contiguity: Capillary water enters exclusively from the bottom reservoir.
2. Column Monotonicity: Water absorption front moves upward; non-contiguous surface droplets
   or reflection specularities are filtered out.
3. Wetted Area Ratio (WAR): Ratio of wetted pixels to specimen area (or total area).
"""

from typing import Optional, Tuple, Dict
import cv2
import numpy as np


def enforce_waterline_physics(
    binary_mask: np.ndarray,
    bottom_margin: int = 32,
    close_kernel_size: Tuple[int, int] = (5, 7),
) -> np.ndarray:
    """Enforces physical sorptivity boundary constraints on a binary mask.

    Args:
        binary_mask: 2D binary numpy array (0 or 1, or 0 or 255).
        bottom_margin: Pixel tolerance for bottom reservoir contiguity.
        close_kernel_size: Kernel dimensions for morphological closing.

    Returns:
        Filtered 2D binary uint8 mask (0 and 1).
    """
    mask = (binary_mask > 0).astype(np.uint8)
    H, W = mask.shape
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return np.zeros_like(mask)

    max_y_reached = max(
        stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT] - 1
        for i in range(1, num_labels)
    )

    bottom_connected_mask = np.zeros_like(mask)
    for i in range(1, num_labels):
        comp_bottom = stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT] - 1
        if comp_bottom >= max(H - 1 - bottom_margin, max_y_reached - bottom_margin):
            bottom_connected_mask[labels == i] = 1

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, close_kernel_size)
    closed = cv2.morphologyEx(bottom_connected_mask, cv2.MORPH_CLOSE, kernel)

    final_mask = np.zeros_like(closed)
    for col in range(W):
        wet_rows = np.where(closed[:, col] == 1)[0]
        if len(wet_rows) == 0:
            continue
        bottom_pixel = wet_rows[-1]
        waterline = bottom_pixel
        while waterline >= 0 and closed[waterline, col] == 1:
            waterline -= 1
        waterline += 1
        final_mask[waterline : bottom_pixel + 1, col] = 1

    return final_mask.astype(np.uint8)


def compute_war(
    mask: np.ndarray,
    roi_mask: Optional[np.ndarray] = None
) -> float:
    """Calculates Wetted Area Ratio: |M_wet ∩ M_ROI| / |M_ROI| (or / total_pixels).

    Args:
        mask: 2D binary array of wetted region.
        roi_mask: Optional 2D binary array of cementitious specimen ROI.

    Returns:
        Float WAR ratio in [0.0, 1.0].
    """
    mask_bin = (mask > 0)
    if roi_mask is not None:
        roi_bin = (roi_mask > 0)
        roi_count = int(np.sum(roi_bin))
        if roi_count == 0:
            return 0.0
        wetted_in_roi = int(np.sum(mask_bin & roi_bin))
        return float(wetted_in_roi / roi_count)

    total_pixels = max(mask.size, 1)
    return float(np.sum(mask_bin) / total_pixels)


def compute_metrics(
    pred_binary: np.ndarray,
    gt_binary: np.ndarray,
    eps: float = 1e-7
) -> Dict[str, float]:
    """Computes standard segmentation overlap metrics when ground truth is provided."""
    pred = (pred_binary > 0).astype(np.float32)
    gt = (gt_binary > 0).astype(np.float32)

    tp = float((pred * gt).sum())
    fp = float((pred * (1.0 - gt)).sum())
    fn = float(((1.0 - pred) * gt).sum())
    tn = float(((1.0 - pred) * (1.0 - gt)).sum())

    iou = (tp + eps) / (tp + fp + fn + eps)
    dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    accuracy = (tp + tn) / (tp + fp + fn + tn + eps)

    return {
        "iou": float(iou),
        "dice": float(dice),
        "precision": float(precision),
        "recall": float(recall),
        "accuracy": float(accuracy),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn)
    }
