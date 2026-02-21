"""Masked MUA reconstruction pretraining model.

Wraps NeuroBridgeEncoder + MaskingDecoder for self-supervised pretraining
on TVSD normMUA data using electrode masking.

Pipeline:
  Input MUA → Mask electrodes → Encoder (on unmasked) → Decoder → Reconstruct masked MUA
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

from neurobridge.models.neurobridge_encoder import NeuroBridgeEncoder
from neurobridge.pretraining.masking_decoder import MaskingDecoder
from neurobridge.pretraining.masking_strategy import MaskingStrategy


class MaskingPretrainingModel(nn.Module):
    """Full encoder-decoder model for masked MUA reconstruction.

    Args:
        encoder_dim: Encoder hidden dimension
        encoder_depth: Number of encoder self-attention layers
        encoder_dim_head: Encoder attention head dimension
        encoder_cross_heads: Encoder cross-attention heads
        encoder_self_heads: Encoder self-attention heads
        num_latents: Number of latent tokens
        decoder_heads: Decoder cross-attention heads
        decoder_dim_head: Decoder attention head dimension
        mask_ratio: Fraction of electrodes to mask
    """

    def __init__(
        self,
        encoder_dim: int = 128,
        encoder_depth: int = 6,
        encoder_dim_head: int = 64,
        encoder_cross_heads: int = 2,
        encoder_self_heads: int = 4,
        num_latents: int = 8,
        decoder_heads: int = 2,
        decoder_dim_head: int = 64,
        mask_ratio: float = 0.25,
    ):
        super().__init__()

        self.encoder = NeuroBridgeEncoder(
            dim=encoder_dim,
            depth=encoder_depth,
            dim_head=encoder_dim_head,
            cross_heads=encoder_cross_heads,
            self_heads=encoder_self_heads,
            num_latents=num_latents,
        )

        self.decoder = MaskingDecoder(
            dim=encoder_dim,
            n_heads=decoder_heads,
            dim_head=decoder_dim_head,
        )

        self.masking = MaskingStrategy(
            electrode_mask_ratio=mask_ratio,
        )

    def forward(
        self,
        input_unit_index: torch.Tensor,
        input_timestamps: torch.Tensor,
        input_values: torch.Tensor,
        input_mask: torch.Tensor,
        latent_index: torch.Tensor,
        latent_timestamps: torch.Tensor,
        region_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with masking, encoding, and reconstruction.

        Returns dict with:
            - predictions: (batch, n_masked, 1) predicted MUA at masked positions
            - targets: (batch, n_masked, 1) actual MUA at masked positions
            - loss: scalar MSE loss on masked positions
            - n_masked: number of masked positions per sample
        """
        # Step 1: Apply masking
        mask_result = self.masking(
            input_values=input_values,
            input_unit_index=input_unit_index,
            input_timestamps=input_timestamps,
            input_mask=input_mask,
            region_labels=region_labels,
        )

        masked_values = mask_result["masked_input_values"]
        masked_mask = mask_result["masked_input_mask"]
        mask_positions = mask_result["mask_positions"]
        original_values = mask_result["original_values"]

        # Step 2: Encode (only unmasked tokens visible to encoder)
        latents = self.encoder(
            input_unit_index=input_unit_index,
            input_timestamps=input_timestamps,
            input_values=masked_values,
            input_mask=masked_mask,
            latent_index=latent_index,
            latent_timestamps=latent_timestamps,
        )

        # Step 3: Decode ALL positions (both masked and unmasked)
        # The loss is computed only on masked positions
        predictions = self.decoder(
            encoder_latents=latents,
            latent_timestamps=latent_timestamps,
            query_unit_index=input_unit_index,
            query_timestamps=input_timestamps,
        )

        # Step 4: Compute loss on masked positions only
        loss = self._compute_masked_loss(
            predictions, original_values, mask_positions
        )

        # Also compute loss on all positions (for monitoring)
        all_loss = F.mse_loss(
            predictions[input_mask.unsqueeze(-1).expand_as(predictions)],
            original_values[input_mask.unsqueeze(-1).expand_as(original_values)],
        )

        return {
            "loss": loss,
            "all_loss": all_loss,
            "n_masked": mask_positions.sum().item(),
            "n_total": input_mask.sum().item(),
            "mask_ratio_actual": mask_positions.float().mean().item(),
        }

    def _compute_masked_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask_positions: torch.Tensor,
    ) -> torch.Tensor:
        """Compute MSE loss only on masked positions.

        Args:
            predictions: (batch, n_electrodes, 1)
            targets: (batch, n_electrodes, 1)
            mask_positions: (batch, n_electrodes) boolean

        Returns:
            Scalar MSE loss
        """
        # Expand mask to match value dimension
        mask_3d = mask_positions.unsqueeze(-1).expand_as(predictions)

        pred_masked = predictions[mask_3d]
        target_masked = targets[mask_3d]

        if pred_masked.numel() == 0:
            return torch.tensor(0.0, device=predictions.device, requires_grad=True)

        return F.mse_loss(pred_masked, target_masked)

    def encode(
        self,
        input_unit_index: torch.Tensor,
        input_timestamps: torch.Tensor,
        input_values: torch.Tensor,
        input_mask: torch.Tensor,
        latent_index: torch.Tensor,
        latent_timestamps: torch.Tensor,
    ) -> torch.Tensor:
        """Encode without masking (for downstream use after pretraining).

        Returns: (batch, num_latents, dim) latent representation
        """
        return self.encoder(
            input_unit_index=input_unit_index,
            input_timestamps=input_timestamps,
            input_values=input_values,
            input_mask=input_mask,
            latent_index=latent_index,
            latent_timestamps=latent_timestamps,
        )

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
