"""Masking decoder for self-supervised MUA reconstruction.

Cross-attends from grid query tokens to encoder latents to reconstruct
masked electrode MUA values.
"""

import torch
import torch.nn as nn
from typing import Optional

from torch_brain.nn import (
    Embedding,
    FeedForward,
    InfiniteVocabEmbedding,
    RotaryCrossAttention,
    RotaryTimeEmbedding,
)


class MaskingDecoder(nn.Module):
    """Decoder for masked MUA reconstruction.

    Uses cross-attention from learnable query tokens (one per electrode)
    to encoder latent tokens, then projects to scalar MUA predictions.

    Architecture:
        Query tokens (unit_emb + positional) → Cross-Attention ← Encoder latents
        → FFN → Linear → predicted MUA values

    Args:
        dim: Model dimension (must match encoder)
        n_heads: Number of cross-attention heads
        dim_head: Attention head dimension
        ffn_dropout: Feed-forward dropout
        atn_dropout: Attention dropout
        emb_init_scale: Embedding initialization scale
    """

    def __init__(
        self,
        dim: int = 128,
        n_heads: int = 2,
        dim_head: int = 64,
        ffn_dropout: float = 0.2,
        atn_dropout: float = 0.0,
        emb_init_scale: float = 0.02,
        t_min: float = 1e-4,
        t_max: float = 2.0627,
    ):
        super().__init__()

        self.dim = dim

        # Decoder query embeddings (shared with encoder's unit embeddings)
        # We'll use the same InfiniteVocabEmbedding for electrode identity
        self.query_unit_emb = InfiniteVocabEmbedding(dim // 2, init_scale=emb_init_scale)

        # Value token for queries (initialized to zero — decoder queries have no value info)
        self.query_value_proj = nn.Linear(1, dim // 2)
        nn.init.zeros_(self.query_value_proj.weight)
        nn.init.zeros_(self.query_value_proj.bias)

        # Rotary time embedding
        self.rotary_emb = RotaryTimeEmbedding(
            head_dim=dim_head,
            rotate_dim=dim_head // 2,
            t_min=t_min,
            t_max=t_max,
        )

        # Cross-attention: queries attend to encoder latents
        self.cross_attn = RotaryCrossAttention(
            dim=dim,
            heads=n_heads,
            dropout=atn_dropout,
            dim_head=dim_head,
            rotate_value=True,
        )
        self.cross_ffn = nn.Sequential(
            nn.LayerNorm(dim),
            FeedForward(dim=dim, dropout=ffn_dropout),
        )

        # Output projection: dim → 1 (predict MUA value)
        self.output_norm = nn.LayerNorm(dim)
        self.output_proj = nn.Linear(dim, 1)
        nn.init.zeros_(self.output_proj.bias)

    def forward(
        self,
        encoder_latents: torch.Tensor,
        latent_timestamps: torch.Tensor,
        query_unit_index: torch.Tensor,
        query_timestamps: torch.Tensor,
    ) -> torch.Tensor:
        """Reconstruct MUA values from encoder latents.

        Args:
            encoder_latents: (batch, n_latent, dim) from NeuroBridgeEncoder
            latent_timestamps: (batch, n_latent) latent token timestamps
            query_unit_index: (batch, n_query) electrode indices for reconstruction
            query_timestamps: (batch, n_query) timestamps for reconstruction

        Returns:
            (batch, n_query, 1) predicted MUA values
        """
        batch_size = encoder_latents.shape[0]
        n_query = query_unit_index.shape[1]

        # Build query tokens from electrode embeddings
        # Use zeros as "value" since we're predicting, not conditioning
        dummy_values = torch.zeros(
            batch_size, n_query, 1,
            device=encoder_latents.device, dtype=encoder_latents.dtype,
        )
        query_tokens = torch.cat(
            (self.query_value_proj(dummy_values), self.query_unit_emb(query_unit_index)),
            dim=-1,
        )

        # Positional embeddings
        query_pos = self.rotary_emb(query_timestamps)
        latent_pos = self.rotary_emb(latent_timestamps)

        # Cross-attention: queries attend to latents
        decoded = query_tokens + self.cross_attn(
            query_tokens,
            encoder_latents,
            query_pos,
            latent_pos,
        )
        decoded = decoded + self.cross_ffn(decoded)

        # Project to MUA values
        decoded = self.output_norm(decoded)
        predictions = self.output_proj(decoded)  # (batch, n_query, 1)

        return predictions

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
