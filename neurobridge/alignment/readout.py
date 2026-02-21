"""Neural readout module: cross-attention from encoder latents to learnable queries.

Extracts a fixed-size representation from the encoder's variable latent tokens
using K learnable query tokens that cross-attend to the latents.
"""

import torch
import torch.nn as nn
from torch_brain.nn import (
    FeedForward,
    RotaryCrossAttention,
    RotaryTimeEmbedding,
)


class NeuralReadout(nn.Module):
    """Learnable readout from encoder latent tokens.

    Uses K learnable query tokens that cross-attend to the encoder's latent tokens,
    producing a fixed-size representation suitable for projection to CLIP space.

    Args:
        dim: Model dimension (must match encoder dim)
        n_queries: Number of learnable readout queries (default: 8)
        n_heads: Number of cross-attention heads (default: 2)
        dim_head: Attention head dimension (default: 64)
        dropout: Attention dropout (default: 0.0)
        ffn_dropout: FFN dropout (default: 0.2)
    """

    def __init__(
        self,
        dim: int = 128,
        n_queries: int = 8,
        n_heads: int = 2,
        dim_head: int = 64,
        dropout: float = 0.0,
        ffn_dropout: float = 0.2,
    ):
        super().__init__()

        self.dim = dim
        self.n_queries = n_queries

        # Learnable query tokens
        self.queries = nn.Parameter(torch.randn(n_queries, dim) * 0.02)

        # Cross-attention: queries attend to latents
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            FeedForward(dim=dim, dropout=ffn_dropout),
        )

    def forward(
        self, latents: torch.Tensor
    ) -> torch.Tensor:
        """Read out from encoder latent tokens.

        Args:
            latents: Encoder output (batch, n_latent, dim)

        Returns:
            Readout tokens (batch, n_queries, dim)
        """
        batch_size = latents.shape[0]

        # Expand queries for batch
        queries = self.queries.unsqueeze(0).expand(batch_size, -1, -1)

        # Cross-attention: queries attend to encoder latents
        attended, _ = self.cross_attn(
            query=self.norm1(queries),
            key=latents,
            value=latents,
        )
        queries = queries + attended
        queries = queries + self.ffn(queries)

        return queries
