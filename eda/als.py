"""Adaptive Layer Selection (ALS)."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveLayerSelection(nn.Module):
    """Select pivotal layers with distance-aware Gumbel-Softmax and STE."""

    def __init__(
        self,
        num_layers: int,
        temperature_init: float = 5.0,
        temperature_min: float = 0.5,
        use_ste: bool = True,
        ste_threshold: float = 0.5,
        distance_metric: str = "l2",
    ):
        super().__init__()
        # Paper notation: phi_l, the intrinsic learnable layer preference.
        self.intrinsic_logits = nn.Parameter(torch.zeros(num_layers))
        self.temperature_init = temperature_init
        self.temperature_min = temperature_min
        self.use_ste = use_ste
        self.ste_threshold = ste_threshold
        self.distance_metric = distance_metric
        self._progress = 0.0

    def set_progress(self, progress: float) -> None:
        self._progress = max(0.0, min(1.0, float(progress)))

    def get_temperature(self) -> float:
        # Exponential interpolation reaches both configured endpoints exactly:
        # tau(0) = temperature_init and tau(1) = temperature_min.
        if self.temperature_init <= 0 or self.temperature_min <= 0:
            raise ValueError("ALS temperatures must be positive")
        ratio = self.temperature_min / self.temperature_init
        return self.temperature_init * math.exp(
            math.log(ratio) * self._progress
        )

    def forward(
        self,
        training: bool = True,
        source_activations: Optional[torch.Tensor] = None,
        target_activations: Optional[torch.Tensor] = None,
        distance_weight: float = 0.0,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not training:
            return None, None

        selection_logits = self.intrinsic_logits
        if (
            distance_weight > 0
            and source_activations is not None
            and target_activations is not None
        ):
            distances = self.compute_distance(
                source_activations,
                target_activations,
                self.distance_metric,
            )
            # Paper Eq. 4: z_l = phi_l + lambda * d_l.
            selection_logits = selection_logits + distance_weight * distances

        probabilities = F.gumbel_softmax(
            selection_logits,
            tau=self.get_temperature(),
            hard=False,
            dim=0,
        )

        if self.use_ste:
            hard_mask = (probabilities > self.ste_threshold).float()
            if not hard_mask.any():
                # A thresholded STE can otherwise select an argmax whose
                # forward weight is still zero. Make the fallback a genuine
                # one-hot selection while retaining soft gradients.
                fallback_index = torch.argmax(probabilities)
                hard_mask = hard_mask.clone()
                hard_mask[fallback_index] = 1.0
            ste_probabilities = hard_mask + (
                probabilities - probabilities.detach()
            )
            selected_indices = hard_mask.nonzero(as_tuple=False).squeeze(-1)
            weights = ste_probabilities[selected_indices]
        else:
            selected_indices = torch.arange(
                len(probabilities), device=probabilities.device
            )
            weights = probabilities

        return selected_indices, weights

    def compute_distance(
        self,
        source_activations: torch.Tensor,
        target_activations: torch.Tensor,
        distance_metric: str = "l2",
    ) -> torch.Tensor:
        """Compute the normalized source-target distance for every layer."""

        source_mean = source_activations.mean(0)
        target_mean = target_activations.mean(0)

        if distance_metric == "l2":
            distances = torch.norm(source_mean - target_mean, p=2, dim=-1)
        elif distance_metric == "l1":
            distances = torch.norm(source_mean - target_mean, p=1, dim=-1)
        elif distance_metric == "cosine":
            source_normalized = F.normalize(source_mean, p=2, dim=-1)
            target_normalized = F.normalize(target_mean, p=2, dim=-1)
            distances = 1.0 - (
                source_normalized * target_normalized
            ).sum(dim=-1)
        elif distance_metric == "mmd":
            distances = torch.norm(
                source_mean - target_mean, p=2, dim=-1
            ).pow(2)
        elif distance_metric == "coral":
            num_layers = source_activations.size(1)
            distances = torch.zeros(
                num_layers, device=source_activations.device
            )
            for layer_index in range(num_layers):
                source_layer = source_activations[:, layer_index, :]
                target_layer = target_activations[:, layer_index, :]
                source_centered = source_layer - source_layer.mean(
                    0, keepdim=True
                )
                target_centered = target_layer - target_layer.mean(
                    0, keepdim=True
                )
                source_covariance = (
                    source_centered.T @ source_centered
                ) / max(1, source_layer.size(0) - 1)
                target_covariance = (
                    target_centered.T @ target_centered
                ) / max(1, target_layer.size(0) - 1)
                distances[layer_index] = torch.norm(
                    source_covariance - target_covariance, p="fro"
                )
        else:
            raise ValueError(f"Unknown distance metric: {distance_metric}")

        return (distances - distances.min()) / (
            distances.max() - distances.min() + 1e-8
        )
