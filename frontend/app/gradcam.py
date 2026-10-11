"""Grad-CAM++ Explainable AI (XAI) engine for U-Net++ segmentation network.

Faithfully implements the analytical weighting formulation of Grad-CAM++:
Chattopadhay et al., 'Grad-CAM++: Generalized Gradient-Based Visual Explanations for Deep Convolutional Networks', 2018.
"""

from typing import Tuple, Optional, Callable, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAMPlusPlus:
    """Analytical Grad-CAM++ implementation for semantic segmentation architectures."""

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None

        self.fwd_handle = target_layer.register_forward_hook(self._save_activations)
        self.bwd_handle = target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, inp, out):
        self.activations = out

    def _save_gradients(self, module, grad_inp, grad_out):
        self.gradients = grad_out[0]

    def remove_hooks(self):
        """Cleanly removes PyTorch forward and backward hooks."""
        if hasattr(self, "fwd_handle") and self.fwd_handle:
            self.fwd_handle.remove()
        if hasattr(self, "bwd_handle") and self.bwd_handle:
            self.bwd_handle.remove()

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_scalar_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        pred_threshold: float = 0.50
    ) -> Tuple[np.ndarray, torch.Tensor]:
        """Generates the 2D Grad-CAM++ attribution heatmap.

        Args:
            input_tensor: 4D Tensor (1, 3, H, W).
            target_scalar_fn: Function mapping logits (1, 1, H, W) -> scalar loss.
            pred_threshold: Probability threshold for positive segmentation.

        Returns:
            Tuple of (cam_normalized [H, W], logits [1, 1, H, W]).
        """
        self.model.zero_grad()
        if not input_tensor.requires_grad:
            input_tensor = input_tensor.clone().detach().requires_grad_(True)

        logits = self.model(input_tensor)

        if target_scalar_fn is None:
            probs = torch.sigmoid(logits)
            pred_mask = (probs > pred_threshold).float()
            if pred_mask.sum() > 0:
                target_scalar = (logits * pred_mask).sum() / pred_mask.sum()
            else:
                target_scalar = torch.topk(logits.flatten(), k=min(100, logits.numel())).values.mean()
        else:
            target_scalar = target_scalar_fn(logits)

        target_scalar.backward(retain_graph=True)

        grads = self.gradients
        acts = self.activations

        if grads is None or acts is None:
            raise RuntimeError("Gradients or activations were not captured. Check target_layer hook registration.")

        # Analytical Grad-CAM++ Alpha Weighting
        grads_power_2 = grads ** 2
        grads_power_3 = grads ** 3
        sum_acts = torch.sum(acts * grads_power_3, dim=(-2, -1), keepdim=True)
        eps = 1e-7
        aij = grads_power_2 / (2 * grads_power_2 + sum_acts + eps)
        aij = torch.where(grads != 0, aij, torch.zeros_like(aij))

        weights = torch.sum(aij * torch.relu(grads), dim=(-2, -1), keepdim=True)
        cam = torch.sum(weights * acts, dim=1, keepdim=True)
        cam = torch.relu(cam)

        # Bilinear upsample to input resolution
        cam = F.interpolate(cam, size=input_tensor.shape[-2:], mode="bilinear", align_corners=False)
        cam_np = cam[0, 0].detach().cpu().numpy()

        c_min, c_max = float(cam_np.min()), float(cam_np.max())
        if c_max > c_min:
            cam_norm = (cam_np - c_min) / (c_max - c_min)
        else:
            cam_norm = np.zeros_like(cam_np)

        return cam_norm, logits


def get_target_layer_by_name(model: nn.Module, layer_name: str) -> nn.Module:
    """Resolves string layer designation to the matching submodule in U-Net++."""
    layer_map: Dict[str, nn.Module] = {
        "decoder_x_0_4": model.decoder.blocks["x_0_4"],
        "decoder_x_0_3": model.decoder.blocks["x_0_3"],
        "encoder_layer4": model.encoder.layer4[-1],
    }
    if layer_name in layer_map:
        return layer_map[layer_name]

    # Default to primary late decoder block x_0_4
    return model.decoder.blocks["x_0_4"]
