"""NeuroBridge Encoder: CaPOYO-based neural encoder for TVSD MUA data.

Based on the CaPOYO architecture but adapted for:
1. TVSD MUA continuous value tokenization
2. Neural representation extraction (no behavioral readout)
3. Future CLIP alignment (produces fixed-size latent embeddings)

Architecture:
  MUA values → input_value_map (Linear 1→dim//2)
             + unit_emb (electrode ID → dim//2)
             → concat → dim-dimensional tokens
  → RoPE Cross-Attention (1024 tokens → num_latents latent tokens)
  → Self-Attention × depth
  → Latent representation (num_latents × dim)
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import cat, tensor, int64
from einops import rearrange, repeat
from torchtyping import TensorType

from torch_brain.nn import (
    Embedding,
    FeedForward,
    InfiniteVocabEmbedding,
    RotaryCrossAttention,
    RotarySelfAttention,
    RotaryTimeEmbedding,
)
from torch_brain.data import pad8, track_mask8


class NeuroBridgeEncoder(nn.Module):
    """Neural encoder for MUA data using PerceiverIO architecture.

    Encodes 1024-channel MUA data into a fixed-size latent representation
    that can be projected to CLIP space for image reconstruction.

    Args:
        dim: Model dimension (default: 128)
        depth: Number of self-attention layers (default: 6)
        dim_head: Attention head dimension (default: 64)
        cross_heads: Number of cross-attention heads (default: 2)
        self_heads: Number of self-attention heads (default: 4)
        num_latents: Number of latent tokens (default: 8)
        ffn_dropout: Feed-forward dropout (default: 0.2)
        lin_dropout: Linear dropout (default: 0.4)
        atn_dropout: Attention dropout (default: 0.0)
        emb_init_scale: Embedding initialization scale (default: 0.02)
    """

    def __init__(
        self,
        *,
        dim: int = 128,
        depth: int = 6,
        dim_head: int = 64,
        cross_heads: int = 2,
        self_heads: int = 4,
        num_latents: int = 8,
        ffn_dropout: float = 0.2,
        lin_dropout: float = 0.4,
        atn_dropout: float = 0.0,
        emb_init_scale: float = 0.02,
        t_min: float = 1e-4,
        t_max: float = 2.0627,
    ):
        super().__init__()

        self.dim = dim
        self.depth = depth
        self.num_latents = num_latents

        # Input value map: continuous MUA value → dim//2
        self.input_value_map = nn.Linear(1, dim // 2)
        nn.init.trunc_normal_(self.input_value_map.weight, 0, emb_init_scale)
        nn.init.zeros_(self.input_value_map.bias)

        # Embeddings
        self.unit_emb = InfiniteVocabEmbedding(dim // 2, init_scale=emb_init_scale)
        self.latent_emb = Embedding(num_latents, dim, init_scale=emb_init_scale)
        self.rotary_emb = RotaryTimeEmbedding(
            head_dim=dim_head,
            rotate_dim=dim_head // 2,
            t_min=t_min,
            t_max=t_max,
        )

        self.dropout = nn.Dropout(p=lin_dropout)

        # Encoder: cross-attention from inputs to latents
        self.enc_atn = RotaryCrossAttention(
            dim=dim,
            heads=cross_heads,
            dropout=atn_dropout,
            dim_head=dim_head,
            rotate_value=True,
        )
        self.enc_ffn = nn.Sequential(
            nn.LayerNorm(dim), FeedForward(dim=dim, dropout=ffn_dropout)
        )

        # Processor: self-attention on latents
        self.proc_layers = nn.ModuleList([])
        for _ in range(depth):
            self.proc_layers.append(
                nn.ModuleList(
                    [
                        RotarySelfAttention(
                            dim=dim,
                            heads=self_heads,
                            dropout=atn_dropout,
                            dim_head=dim_head,
                            rotate_value=True,
                        ),
                        nn.Sequential(
                            nn.LayerNorm(dim),
                            FeedForward(dim=dim, dropout=ffn_dropout),
                        ),
                    ]
                )
            )

        # Final layer norm on latent output
        self.latent_norm = nn.LayerNorm(dim)

    def forward(
        self,
        *,
        input_unit_index: TensorType["batch", "n_in"],
        input_timestamps: TensorType["batch", "n_in"],
        input_values: TensorType["batch", "n_in", 1],
        input_mask: Optional[TensorType["batch", "n_in"]] = None,
        latent_index: TensorType["batch", "n_latent"],
        latent_timestamps: TensorType["batch", "n_latent"],
    ) -> TensorType["batch", "n_latent", "dim"]:
        """Encode MUA data into latent representations.

        Returns:
            Latent tensor of shape (batch, num_latents, dim)
        """
        if self.unit_emb.is_lazy():
            raise ValueError(
                "Unit vocabulary not initialized. "
                "Call model.unit_emb.initialize_vocab(unit_ids) first."
            )

        # Input embedding: value_map + unit_emb → dim
        inputs = cat(
            (self.input_value_map(input_values), self.unit_emb(input_unit_index)),
            dim=-1,
        )
        input_timestamp_emb = self.rotary_emb(input_timestamps)

        # Latent tokens
        latents = self.latent_emb(latent_index)
        latent_timestamp_emb = self.rotary_emb(latent_timestamps)

        # Encode: cross-attention (latents attend to inputs)
        latents = latents + self.enc_atn(
            latents,
            inputs,
            latent_timestamp_emb,
            input_timestamp_emb,
            input_mask,
        )
        latents = latents + self.enc_ffn(latents)

        # Process: self-attention on latents
        for self_attn, self_ff in self.proc_layers:
            latents = latents + self.dropout(self_attn(latents, latent_timestamp_emb))
            latents = latents + self.dropout(self_ff(latents))

        # Normalize
        latents = self.latent_norm(latents)

        return latents

    def get_latent_representation(
        self, latents: TensorType["batch", "n_latent", "dim"]
    ) -> TensorType["batch", "dim"]:
        """Pool latent tokens into a single representation vector.

        Uses mean pooling over latent tokens.
        """
        return latents.mean(dim=1)

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
