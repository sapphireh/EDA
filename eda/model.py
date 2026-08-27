"""Top-level Elastic Domain Adaptation model wrapper."""

from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn as nn

from .als import AdaptiveLayerSelection
from .esm import ElasticShiftModule


class EDAModel(nn.Module):
    """Compose an LLM with Adaptive Layer Selection and Elastic Shift."""

    def __init__(
        self,
        base_model: nn.Module,
        domain_loss_weight: float = 0.2,
        activation_pooling: str = "mean",
        grl_gamma: float = 10.0,
        num_target_domains: int = 8,
        # ALS
        enable_als: bool = True,
        als_temperature_init: float = 5.0,
        als_temperature_min: float = 0.5,
        als_use_ste: bool = True,
        als_ste_threshold: float = 0.5,
        als_distance_weight: float = 0.0,
        als_distance_metric: str = "l2",
        # ESM adapters
        esm_use_adapter: bool = False,
        esm_adapter_hidden_dim: int = 1024,
        esm_adapter_depth: int = 15,
        esm_adapter_dropout: float = 0.1,
        esm_shift_scale: float = 0.5,
        esm_adapter_use_gate: bool = False,
        esm_adapter_spectral_norm: bool = False,
        # ESM fusion
        esm_use_fusion: bool = False,
        esm_fusion_method: str = "film",
        esm_fusion_hidden_dim: int = 1024,
        # ESM discriminators
        esm_discriminator_hidden_dim: int = 1024,
        esm_discriminator_depth: int = 15,
    ):
        super().__init__()
        self.base_model = base_model

        # EDAModel is intentionally a lightweight nn.Module wrapper rather
        # than a PreTrainedModel subclass.  Explicitly expose the attributes
        # that Transformers uses for generation and device placement.  In
        # particular, hiding hf_device_map made Trainer move/model-replicate
        # an already sharded QLoRA model and caused cross-device failures.
        for attribute in (
            "config",
            "generation_config",
            "hf_device_map",
            "quantization_method",
            "is_loaded_in_4bit",
            "is_loaded_in_8bit",
        ):
            value = getattr(base_model, attribute, None)
            if value is not None:
                setattr(self, attribute, value)
        self.domain_loss_weight = domain_loss_weight
        self.activation_pooling = activation_pooling
        self.grl_gamma = grl_gamma
        self.num_target_domains = int(max(1, num_target_domains))
        self._progress = 0.0

        self.enable_als = enable_als
        self.als_distance_weight = als_distance_weight
        self.adaptive_layer_selection: Optional[
            AdaptiveLayerSelection
        ] = None
        self._als_parameters = {
            "temperature_init": als_temperature_init,
            "temperature_min": als_temperature_min,
            "use_ste": als_use_ste,
            "ste_threshold": als_ste_threshold,
            "distance_metric": als_distance_metric,
        }

        self.elastic_shift_module: Optional[ElasticShiftModule] = None
        self._esm_parameters = {
            "discriminator_hidden_dimension": esm_discriminator_hidden_dim,
            "discriminator_depth": esm_discriminator_depth,
            "use_adapter": esm_use_adapter,
            "adapter_hidden_dimension": esm_adapter_hidden_dim,
            "adapter_depth": esm_adapter_depth,
            "adapter_dropout": esm_adapter_dropout,
            "shift_scale": esm_shift_scale,
            "adapter_use_gate": esm_adapter_use_gate,
            "adapter_spectral_norm": esm_adapter_spectral_norm,
            "use_fusion": esm_use_fusion,
            "fusion_method": esm_fusion_method,
            "fusion_hidden_dimension": esm_fusion_hidden_dim,
        }

        self._last_total_loss: Optional[float] = None
        self._last_lm_loss: Optional[float] = None
        self._last_domain_loss: Optional[float] = None
        self._last_selected_layers: Optional[List[int]] = None
        self._last_layer_weights: Optional[List[float]] = None
        self._last_layer_scores: Optional[torch.Tensor] = None
        self.pad_token_id: int = 0

    def set_progress(self, progress: float) -> None:
        self._progress = max(0.0, min(1.0, float(progress)))
        if self.adaptive_layer_selection is not None:
            self.adaptive_layer_selection.set_progress(progress)

    def compute_grl_alpha(self) -> float:
        return 2.0 / (
            1.0 + math.exp(-self.grl_gamma * self._progress)
        ) - 1.0

    @torch.no_grad()
    def initialize_eda_modules(
        self,
        hidden_size: int,
        num_layers: int,
        device,
        dtype,
    ) -> None:
        if self.adaptive_layer_selection is None and self.enable_als:
            self.adaptive_layer_selection = AdaptiveLayerSelection(
                num_layers, **self._als_parameters
            ).to(device=device, dtype=dtype)

        if self.elastic_shift_module is None:
            self.elastic_shift_module = ElasticShiftModule(
                hidden_dimension=hidden_size,
                num_target_domains=self.num_target_domains,
                **self._esm_parameters,
            ).to(device=device, dtype=dtype)

    def pool_activation(
        self,
        hidden_state: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.activation_pooling == "mean":
            mask = attention_mask.unsqueeze(-1).to(hidden_state.dtype)
            summed = (hidden_state * mask).sum(dim=1)
            denominator = mask.sum(dim=1).clamp(min=1e-6)
            return summed / denominator
        if self.activation_pooling == "bos":
            return hidden_state[:, 0, :]

        lengths = attention_mask.sum(dim=1) - 1
        indices = (
            lengths.clamp(min=0)
            .unsqueeze(-1)
            .unsqueeze(-1)
            .expand(-1, 1, hidden_state.size(-1))
        )
        return hidden_state.gather(1, indices).squeeze(1)

    def extract_layer_activations(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        activations = [
            self.pool_activation(hidden_state, attention_mask)
            for hidden_state in outputs.hidden_states[1:]
        ]
        return torch.stack(activations, dim=1)

    def compute_alignment_loss(
        self,
        source_ids: torch.Tensor,
        source_mask: torch.Tensor,
        target_ids: torch.Tensor,
        target_mask: torch.Tensor,
        target_domain_ids: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        # The source branch is a fixed anchor for both ESM and the
        # distance-aware ALS score.  No domain-alignment gradient consumes
        # its graph, so retaining it wastes a full LLM forward graph.
        with torch.no_grad():
            source_activations = self.extract_layer_activations(
                source_ids, source_mask
            )
        target_activations = self.extract_layer_activations(
            target_ids, target_mask
        )

        num_layers = source_activations.size(1)
        hidden_size = source_activations.size(2)
        device = source_activations.device
        self.initialize_eda_modules(
            hidden_size,
            num_layers,
            device,
            source_activations.dtype,
        )

        source_final_activation = source_activations[:, -1, :]
        target_final_activation = target_activations[:, -1, :]

        if (
            self.enable_als
            and self.adaptive_layer_selection is not None
        ):
            selected_indices, weights = self.adaptive_layer_selection(
                training=True,
                # The source representation is anchored by supervised
                # training. Domain alignment must not update it through the
                # distance-dependent ALS weights.
                source_activations=source_activations.detach(),
                target_activations=target_activations,
                distance_weight=self.als_distance_weight,
            )

            try:
                if selected_indices is not None and weights is not None:
                    self._last_selected_layers = selected_indices.cpu().tolist()
                    self._last_layer_weights = weights.cpu().tolist()
                    with torch.no_grad():
                        scores = (
                            self.adaptive_layer_selection.intrinsic_logits
                        )
                        if self.als_distance_weight > 0:
                            distances = self.adaptive_layer_selection.compute_distance(
                                source_activations,
                                target_activations,
                                self.adaptive_layer_selection.distance_metric,
                            )
                            scores = (
                                scores
                                + self.als_distance_weight * distances
                            )
                        self._last_layer_scores = scores.cpu()
            except Exception:
                pass
        else:
            # The final Llama experiment uses layer 18 for its fixed-layer
            # fallback. It is deliberately preserved for exact behavior.
            selected_indices = torch.tensor(
                [min(18, num_layers - 1)], device=device
            )
            weights = torch.ones(1, device=device)

        if selected_indices is None or self.elastic_shift_module is None:
            return None

        weighted_layer_losses = []
        active_weights = []
        grl_alpha = self.compute_grl_alpha()
        for position, layer_index_tensor in enumerate(selected_indices):
            layer_index = int(layer_index_tensor.item())
            layer_loss = self.elastic_shift_module.compute_layer_domain_loss(
                source_activation=source_activations[:, layer_index, :],
                target_activation=target_activations[:, layer_index, :],
                source_final_activation=source_final_activation,
                target_final_activation=target_final_activation,
                target_domain_ids=target_domain_ids,
                grl_alpha=grl_alpha,
            )
            if layer_loss is not None:
                layer_weight = weights[position]
                weighted_layer_losses.append(layer_weight * layer_loss)
                active_weights.append(layer_weight)

        if not weighted_layer_losses:
            return None
        # Normalize over the active layer set so the domain-loss scale does
        # not jump when STE selects a different number of layers.  Detaching
        # the denominator preserves the selector gradient in the numerator.
        denominator = (
            torch.stack(active_weights).sum().detach().clamp_min(1e-8)
        )
        return torch.stack(weighted_layer_losses).sum() / denominator

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        domain_labels: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        del domain_labels
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=False,
            return_dict=True,
            **kwargs,
        )
        lm_loss = outputs.loss if labels is not None else None
        try:
            if lm_loss is not None:
                self._last_lm_loss = float(
                    lm_loss.detach().float().mean().item()
                )
        except Exception:
            pass
        return {
            "loss": lm_loss,
            "logits": outputs.logits,
            "lm_loss": lm_loss,
        }

    def generate(self, *args, **kwargs):
        kwargs.pop("domain_labels", None)
        kwargs.pop("domain_ids", None)
        kwargs.pop("prompt_lens", None)
        kwargs.pop("labels", None)
        generate = getattr(self.base_model, "generate", None)
        if callable(generate):
            return generate(*args, **kwargs)
        raise AttributeError("Underlying base model has no generate()")
