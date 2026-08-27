"""Elastic Shift Module (ESM) and its domain-adaptation components."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.alpha = alpha
        return inputs.view_as(inputs)

    @staticmethod
    def backward(ctx, gradient: torch.Tensor):
        return -ctx.alpha * gradient, None


class ResidualBlock(nn.Module):
    def __init__(
        self,
        dimension: int,
        hidden_dimension: int,
        dropout: float = 0.1,
        spectral_norm: bool = False,
    ):
        super().__init__()
        linear_1 = nn.Linear(dimension, hidden_dimension)
        linear_2 = nn.Linear(hidden_dimension, dimension)
        if spectral_norm:
            linear_1 = nn.utils.parametrizations.spectral_norm(linear_1)
            linear_2 = nn.utils.parametrizations.spectral_norm(linear_2)
        self.linear_1 = linear_1
        self.linear_2 = linear_2
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.normalization = nn.LayerNorm(dimension)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = self.linear_2(
            self.dropout(self.activation(self.linear_1(inputs)))
        )
        return self.normalization(inputs + residual)


class DomainDiscriminator(nn.Module):
    """Binary source-target discriminator for one perturbation domain."""

    def __init__(
        self,
        input_dimension: int,
        hidden_dimension: int = 1024,
        output_dimension: int = 1,
        depth: int = 15,
        dropout: float = 0.1,
        spectral_norm: bool = False,
    ):
        super().__init__()
        input_projection = nn.Linear(input_dimension, hidden_dimension)
        if spectral_norm:
            input_projection = nn.utils.parametrizations.spectral_norm(
                input_projection
            )
        self.input_projection = input_projection
        self.blocks = nn.Sequential(
            *[
                ResidualBlock(
                    hidden_dimension,
                    hidden_dimension,
                    dropout,
                    spectral_norm,
                )
                for _ in range(max(1, depth - 1))
            ]
        )
        output = nn.Linear(hidden_dimension, output_dimension)
        if spectral_norm:
            output = nn.utils.parametrizations.spectral_norm(output)
        self.output = output

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.output(
            self.blocks(self.input_projection(inputs))
        ).squeeze(-1)


class ElasticShiftAdapter(nn.Module):
    """Generate the perturbation-specific residual shift Delta_k."""

    def __init__(
        self,
        input_dimension: int,
        hidden_dimension: int = 1024,
        depth: int = 15,
        dropout: float = 0.1,
        shift_scale: float = 0.5,
        use_gate: bool = False,
        spectral_norm: bool = False,
        gate_hidden_dimension: Optional[int] = None,
    ):
        super().__init__()
        input_projection = nn.Linear(input_dimension, hidden_dimension)
        if spectral_norm:
            input_projection = nn.utils.parametrizations.spectral_norm(
                input_projection
            )
        self.input_projection = input_projection
        self.blocks = nn.Sequential(
            *[
                ResidualBlock(
                    hidden_dimension,
                    hidden_dimension,
                    dropout,
                    spectral_norm,
                )
                for _ in range(max(0, depth))
            ]
        )
        self.output_projection = nn.Linear(hidden_dimension, input_dimension)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

        self.shift_scale = float(shift_scale)
        self.use_gate = bool(use_gate)
        if self.use_gate:
            gate_hidden_dimension = (
                gate_hidden_dimension or input_dimension // 4
            )
            self.gate_projection = nn.Sequential(
                nn.Linear(input_dimension, gate_hidden_dimension),
                nn.GELU(),
                nn.Linear(gate_hidden_dimension, input_dimension),
            )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        hidden = self.input_projection(features)
        hidden = self.blocks(hidden)
        shift = self.output_projection(hidden)
        if self.use_gate:
            gate_scores = torch.sigmoid(self.gate_projection(features))
            shift = shift * gate_scores
        return self.shift_scale * shift


class FiLMFusion(nn.Module):
    """Condition a selected activation on the final-layer activation."""

    def __init__(
        self, hidden_dimension: int, fusion_hidden_dimension: int = 1024
    ):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(hidden_dimension, fusion_hidden_dimension),
            nn.GELU(),
            nn.Linear(fusion_hidden_dimension, hidden_dimension * 2),
        )
        self.normalization = nn.LayerNorm(hidden_dimension)

    def forward(
        self,
        selected_activation: torch.Tensor,
        final_activation: torch.Tensor,
    ) -> torch.Tensor:
        parameters = self.projection(final_activation)
        scale, bias = parameters.chunk(2, dim=-1)
        modulated = (1 + scale) * selected_activation + bias
        return self.normalization(modulated)


class ConcatenationFusion(nn.Module):
    def __init__(
        self, hidden_dimension: int, fusion_hidden_dimension: int = 1024
    ):
        super().__init__()
        # fusion_hidden_dimension is intentionally retained in the public
        # signature to match the original fusion interface.
        del fusion_hidden_dimension
        self.projection = nn.Linear(hidden_dimension * 2, hidden_dimension)
        self.normalization = nn.LayerNorm(hidden_dimension)

    def forward(
        self,
        selected_activation: torch.Tensor,
        final_activation: torch.Tensor,
    ) -> torch.Tensor:
        concatenated = torch.cat(
            [selected_activation, final_activation], dim=-1
        )
        return self.normalization(self.projection(concatenated))


class CrossAttentionFusion(nn.Module):
    def __init__(
        self, hidden_dimension: int, fusion_hidden_dimension: int = 1024
    ):
        super().__init__()
        self.query_projection = nn.Linear(
            hidden_dimension, fusion_hidden_dimension
        )
        self.key_projection = nn.Linear(
            hidden_dimension, fusion_hidden_dimension
        )
        self.value_projection = nn.Linear(
            hidden_dimension, fusion_hidden_dimension
        )
        self.output_projection = nn.Linear(
            fusion_hidden_dimension, hidden_dimension
        )
        self.normalization = nn.LayerNorm(hidden_dimension)
        self.scale = fusion_hidden_dimension**-0.5

    def forward(
        self,
        selected_activation: torch.Tensor,
        final_activation: torch.Tensor,
    ) -> torch.Tensor:
        # Attend over both the selected and final-layer representations.  A
        # one-element key/value sequence would make softmax identically one,
        # leaving the query (and therefore final_activation) without any
        # influence on the result.
        context = torch.stack(
            [selected_activation, final_activation], dim=1
        )
        query = self.query_projection(final_activation).unsqueeze(1)
        key = self.key_projection(context)
        value = self.value_projection(context)
        attention = torch.softmax(
            query @ key.transpose(-2, -1) * self.scale, dim=-1
        )
        output = (attention @ value).squeeze(1)
        return self.normalization(
            selected_activation + self.output_projection(output)
        )


class ElasticShiftModule(nn.Module):
    """Perturbation-aware fusion, elastic shifts, and discrimination."""

    def __init__(
        self,
        hidden_dimension: int,
        num_target_domains: int,
        discriminator_hidden_dimension: int = 1024,
        discriminator_depth: int = 15,
        use_adapter: bool = False,
        adapter_hidden_dimension: int = 1024,
        adapter_depth: int = 15,
        adapter_dropout: float = 0.1,
        shift_scale: float = 0.5,
        adapter_use_gate: bool = False,
        adapter_spectral_norm: bool = False,
        use_fusion: bool = False,
        fusion_method: str = "film",
        fusion_hidden_dimension: int = 1024,
    ):
        super().__init__()
        self.num_target_domains = num_target_domains
        self.use_adapter = use_adapter
        self.use_fusion = use_fusion

        self.domain_discriminators = nn.ModuleList(
            [
                DomainDiscriminator(
                    hidden_dimension,
                    discriminator_hidden_dimension,
                    output_dimension=1,
                    depth=discriminator_depth,
                    dropout=0.1,
                    spectral_norm=False,
                )
                for _ in range(num_target_domains)
            ]
        )

        self.domain_adapters: Optional[nn.ModuleList] = None
        if use_adapter:
            self.domain_adapters = nn.ModuleList(
                [
                    ElasticShiftAdapter(
                        hidden_dimension,
                        adapter_hidden_dimension,
                        adapter_depth,
                        adapter_dropout,
                        shift_scale,
                        adapter_use_gate,
                        adapter_spectral_norm,
                    )
                    for _ in range(num_target_domains)
                ]
            )

        self.fusion: Optional[nn.Module] = None
        if use_fusion:
            if fusion_method == "film":
                self.fusion = FiLMFusion(
                    hidden_dimension, fusion_hidden_dimension
                )
            elif fusion_method == "concat":
                self.fusion = ConcatenationFusion(
                    hidden_dimension, fusion_hidden_dimension
                )
            else:
                self.fusion = CrossAttentionFusion(
                    hidden_dimension, fusion_hidden_dimension
                )

    def fuse(
        self,
        selected_activation: torch.Tensor,
        final_activation: torch.Tensor,
    ) -> torch.Tensor:
        if self.use_fusion and self.fusion is not None:
            return self.fusion(selected_activation, final_activation)
        return selected_activation

    def compute_layer_domain_loss(
        self,
        source_activation: torch.Tensor,
        target_activation: torch.Tensor,
        source_final_activation: torch.Tensor,
        target_final_activation: torch.Tensor,
        target_domain_ids: torch.Tensor,
        grl_alpha: float,
    ) -> Optional[torch.Tensor]:
        # Paper Eq. 9 keeps the supervised source activation as the anchor.
        # Fusion and the elastic shift are target-side transformations only.
        del source_final_activation
        source_features = source_activation.detach()
        target_features = self.fuse(
            target_activation, target_final_activation
        )

        domain_losses = []
        for domain_id in range(1, self.num_target_domains + 1):
            domain_mask = target_domain_ids == domain_id
            if not domain_mask.any():
                continue

            target_domain_features = target_features[domain_mask]
            if self.use_adapter and self.domain_adapters is not None:
                # Paper Eq. 6: Delta_k = Adapter_k(A_l^{t,k}). The adapter
                # consumes the selected raw target activation, while its
                # residual is added to the conditionally fused activation.
                shift = self.domain_adapters[domain_id - 1](
                    target_activation[domain_mask]
                )
                target_domain_features = target_domain_features + shift

            reversed_target_features = GradientReversal.apply(
                target_domain_features, grl_alpha
            )
            discriminator = self.domain_discriminators[domain_id - 1]
            source_logits = discriminator(source_features)
            target_logits = discriminator(reversed_target_features)
            source_loss = F.binary_cross_entropy_with_logits(
                source_logits.float(), torch.zeros_like(source_logits).float()
            )
            target_loss = F.binary_cross_entropy_with_logits(
                target_logits.float(), torch.ones_like(target_logits).float()
            )
            # Keep the two expectations in paper Eq. 9 independently
            # normalized. Concatenating a 5-sample source half with a single
            # target-domain sample would otherwise bias the loss toward source.
            domain_losses.append(0.5 * (source_loss + target_loss))

        if not domain_losses:
            return None
        # A batch may contain a different subset of perturbation domains on
        # every step.  Averaging keeps --domain-loss-weight independent of
        # that incidental domain count.
        return torch.stack(domain_losses).mean()

    def compute_discriminator_pretrain_loss(
        self,
        source_features: torch.Tensor,
        target_features: torch.Tensor,
        target_domain_ids: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        losses = []
        for domain_id in range(1, self.num_target_domains + 1):
            domain_mask = target_domain_ids == domain_id
            if not domain_mask.any():
                continue

            target_domain_features = target_features[domain_mask]
            discriminator = self.domain_discriminators[domain_id - 1]
            source_logits = discriminator(source_features)
            target_logits = discriminator(target_domain_features)
            source_loss = F.binary_cross_entropy_with_logits(
                source_logits.float(), torch.zeros_like(source_logits).float()
            )
            target_loss = F.binary_cross_entropy_with_logits(
                target_logits.float(), torch.ones_like(target_logits).float()
            )
            # Averaging is appropriate for the standalone discriminator
            # initialization objective and keeps its scale domain-count stable.
            losses.append(0.5 * (source_loss + target_loss))

        if not losses:
            return None
        return torch.stack(losses).mean()
