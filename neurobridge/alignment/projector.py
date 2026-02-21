"""Neural projector: maps readout tokens to CLIP embedding space.

3-layer MLP that projects pooled neural readout to the CLIP embedding dimension.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuralProjector(nn.Module):
    """MLP projector from neural readout to CLIP space.

    Takes pooled readout tokens and projects to CLIP embedding dimension
    via a 3-layer MLP with GELU activation and layer normalization.

    Args:
        input_dim: Input dimension (readout dim, default: 128)
        hidden_dim: Hidden layer dimension (default: 512)
        output_dim: Output dimension (CLIP embed dim, default: 768)
        n_queries: Number of readout queries to pool (default: 8)
        pool_method: How to pool readout tokens ('mean', 'concat', 'first')
    """

    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 512,
        output_dim: int = 768,
        n_queries: int = 8,
        pool_method: str = "mean",
    ):
        super().__init__()

        self.pool_method = pool_method

        if pool_method == "concat":
            actual_input_dim = input_dim * n_queries
        else:
            actual_input_dim = input_dim

        self.projector = nn.Sequential(
            nn.Linear(actual_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

        self.output_norm = nn.LayerNorm(output_dim)

    def forward(self, readout_tokens: torch.Tensor) -> torch.Tensor:
        """Project readout tokens to CLIP space.

        Args:
            readout_tokens: (batch, n_queries, dim) from NeuralReadout

        Returns:
            Normalized CLIP-space embedding (batch, output_dim)
        """
        if self.pool_method == "mean":
            pooled = readout_tokens.mean(dim=1)  # (batch, dim)
        elif self.pool_method == "concat":
            pooled = readout_tokens.flatten(start_dim=1)  # (batch, n_queries * dim)
        elif self.pool_method == "first":
            pooled = readout_tokens[:, 0]  # (batch, dim)
        else:
            raise ValueError(f"Unknown pool method: {self.pool_method}")

        projected = self.projector(pooled)  # (batch, output_dim)
        projected = self.output_norm(projected)

        # L2 normalize to match CLIP embedding space
        projected = F.normalize(projected, dim=-1)

        return projected
