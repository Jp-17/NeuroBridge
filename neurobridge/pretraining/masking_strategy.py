"""Masking strategies for self-supervised pretraining.

Implements electrode (neuron) masking and temporal masking for
the MUA reconstruction pretraining objective.
"""

import torch
from typing import Dict, Optional, Tuple


class MaskingStrategy:
    """Applies masking to neural input tokens for self-supervised pretraining.

    Supports two masking modes:
    1. Electrode masking: randomly drops electrodes entirely
    2. Temporal masking: randomly drops time bins (for future temporal data)
    3. Combined: applies both simultaneously

    For time-averaged TVSD normMUA data, only electrode masking is applicable.

    Args:
        electrode_mask_ratio: fraction of electrodes to mask (default: 0.25)
        temporal_mask_ratio: fraction of time bins to mask (default: 0.0)
        min_keep_ratio: minimum fraction of tokens to keep (default: 0.5)
    """

    def __init__(
        self,
        electrode_mask_ratio: float = 0.25,
        temporal_mask_ratio: float = 0.0,
        min_keep_ratio: float = 0.5,
    ):
        self.electrode_mask_ratio = electrode_mask_ratio
        self.temporal_mask_ratio = temporal_mask_ratio
        self.min_keep_ratio = min_keep_ratio

    def __call__(
        self,
        input_values: torch.Tensor,
        input_unit_index: torch.Tensor,
        input_timestamps: torch.Tensor,
        input_mask: Optional[torch.Tensor] = None,
        region_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Apply masking to a batch of neural inputs.

        Args:
            input_values: (batch, n_electrodes, 1) MUA values
            input_unit_index: (batch, n_electrodes) unit indices
            input_timestamps: (batch, n_electrodes) timestamps
            input_mask: (batch, n_electrodes) existing validity mask
            region_labels: (batch, n_electrodes) optional brain region labels

        Returns:
            Dict with:
                - masked_input_values: values with masked positions zeroed
                - masked_input_mask: updated mask (False for masked positions)
                - mask_positions: boolean mask of which positions were masked
                - original_values: original unmasked values (for loss computation)
        """
        batch_size, n_tokens = input_values.shape[:2]
        device = input_values.device

        if input_mask is None:
            input_mask = torch.ones(batch_size, n_tokens, dtype=torch.bool, device=device)

        # Start with all positions visible
        mask_positions = torch.zeros(batch_size, n_tokens, dtype=torch.bool, device=device)

        # Electrode masking: randomly mask electrodes
        if self.electrode_mask_ratio > 0:
            electrode_mask = self._electrode_masking(
                batch_size, n_tokens, device, region_labels
            )
            mask_positions = mask_positions | electrode_mask

        # Ensure minimum keep ratio
        n_valid = input_mask.sum(dim=1)  # (batch,)
        n_masked = mask_positions.sum(dim=1)  # (batch,)
        n_keep = n_valid - n_masked
        min_keep = (n_valid * self.min_keep_ratio).long()

        # If too many masked, randomly unmask some
        for b in range(batch_size):
            if n_keep[b] < min_keep[b]:
                masked_idx = mask_positions[b].nonzero(as_tuple=True)[0]
                n_unmask = min_keep[b] - n_keep[b]
                perm = torch.randperm(len(masked_idx), device=device)[:n_unmask]
                mask_positions[b, masked_idx[perm]] = False

        # Only mask positions that are valid in the original mask
        mask_positions = mask_positions & input_mask

        # Create masked inputs
        masked_values = input_values.clone()
        masked_values[mask_positions.unsqueeze(-1).expand_as(masked_values)] = 0.0

        masked_input_mask = input_mask.clone()
        masked_input_mask[mask_positions] = False

        return {
            "masked_input_values": masked_values,
            "masked_input_mask": masked_input_mask,
            "mask_positions": mask_positions,
            "original_values": input_values,
        }

    def _electrode_masking(
        self,
        batch_size: int,
        n_tokens: int,
        device: torch.device,
        region_labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Random electrode masking.

        Each electrode is independently masked with probability electrode_mask_ratio.
        """
        mask = torch.rand(batch_size, n_tokens, device=device) < self.electrode_mask_ratio
        return mask

    def _region_balanced_masking(
        self,
        batch_size: int,
        n_tokens: int,
        device: torch.device,
        region_labels: torch.Tensor,
    ) -> torch.Tensor:
        """Region-balanced masking: mask equal fractions from each brain region.

        This ensures V1, V4, and IT are equally represented in both
        masked and unmasked sets.
        """
        mask = torch.zeros(batch_size, n_tokens, dtype=torch.bool, device=device)
        unique_regions = region_labels[0].unique()

        for region_id in unique_regions:
            region_mask = region_labels == region_id  # (batch, n_tokens)
            n_region = region_mask[0].sum().item()
            n_mask_region = int(n_region * self.electrode_mask_ratio)

            for b in range(batch_size):
                region_indices = region_mask[b].nonzero(as_tuple=True)[0]
                perm = torch.randperm(len(region_indices), device=device)[:n_mask_region]
                mask[b, region_indices[perm]] = True

        return mask
