"""Stable Diffusion wrapper for neural image reconstruction.

Takes CLIP-space neural embeddings and generates reconstructed images
using Stable Diffusion with the neural embedding as conditioning.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Optional, List


class StableDiffusionWrapper:
    """Wrapper for Stable Diffusion image generation from neural embeddings.

    Uses the neural CLIP embedding as the text encoder output to condition
    Stable Diffusion, replacing the standard text prompt conditioning.

    Args:
        model_id: Stable Diffusion model ID (default: runwayml/stable-diffusion-v1-5)
        device: Device to run on
        cache_dir: Model cache directory
    """

    def __init__(
        self,
        model_id: str = "runwayml/stable-diffusion-v1-5",
        device: str = "cuda",
        cache_dir: Optional[str] = None,
    ):
        self.device = device
        self.model_id = model_id

        from diffusers import StableDiffusionPipeline, DDIMScheduler

        # Load pipeline
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            cache_dir=cache_dir,
        )
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        self.pipe = self.pipe.to(device)
        self.pipe.set_progress_bar_config(disable=True)

        # Get text encoder hidden dimension
        self.text_hidden_dim = self.pipe.text_encoder.config.hidden_size

    @torch.no_grad()
    def generate_from_embedding(
        self,
        neural_clip_embedding: torch.Tensor,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        num_images: int = 1,
        height: int = 512,
        width: int = 512,
        seed: Optional[int] = None,
    ) -> List:
        """Generate images from neural CLIP embeddings.

        Args:
            neural_clip_embedding: (batch, 768) neural embedding in CLIP space
            num_inference_steps: Number of DDIM steps
            guidance_scale: Classifier-free guidance scale
            num_images: Number of images per embedding
            height, width: Output image dimensions
            seed: Random seed

        Returns:
            List of PIL images
        """
        batch_size = neural_clip_embedding.shape[0]
        embed_dim = neural_clip_embedding.shape[1]

        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = None

        # Expand embedding to match expected text encoder output shape
        # SD expects: (batch, seq_len, hidden_dim)
        # We create a sequence of 1 token from our neural embedding
        # and pad to match the expected 77-token sequence length
        prompt_embeds = self._create_prompt_embeds(neural_clip_embedding)

        # Generate
        images = self.pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=torch.zeros_like(prompt_embeds),
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            generator=generator,
        ).images

        return images

    def _create_prompt_embeds(self, neural_embedding: torch.Tensor) -> torch.Tensor:
        """Convert neural CLIP embedding to SD-compatible prompt embeddings.

        Maps (batch, clip_dim) to (batch, 77, text_hidden_dim).
        Uses a simple projection if dimensions don't match.
        """
        batch_size = neural_embedding.shape[0]
        clip_dim = neural_embedding.shape[1]

        # Project to text encoder hidden dim if needed
        if clip_dim != self.text_hidden_dim:
            # Simple linear interpolation/projection
            neural_embedding = neural_embedding.to(torch.float16)
            # For now, zero-pad or truncate
            if clip_dim < self.text_hidden_dim:
                padding = torch.zeros(
                    batch_size, self.text_hidden_dim - clip_dim,
                    device=neural_embedding.device, dtype=neural_embedding.dtype
                )
                neural_embedding = torch.cat([neural_embedding, padding], dim=-1)
            else:
                neural_embedding = neural_embedding[:, :self.text_hidden_dim]

        # Create 77-token sequence (SD's expected sequence length)
        # Put our embedding as the first token, rest are zeros (like padding)
        prompt_embeds = torch.zeros(
            batch_size, 77, self.text_hidden_dim,
            device=neural_embedding.device, dtype=neural_embedding.dtype,
        )
        prompt_embeds[:, 0, :] = neural_embedding

        return prompt_embeds


class DiffusionAdapter(nn.Module):
    """Adapter that expands a single CLIP embedding into SD-compatible conditioning.

    Takes a 768-dim neural CLIP embedding and produces:
    1. A 77-token sequence for text encoder replacement
    2. Optional: a latent noise predictor for img2img-style generation

    Architecture: Token Expander + Refiner
    - Token Expander: Linear(768) → (77, 768) via learned query tokens
    - Refiner: Self-attention to make tokens coherent

    Args:
        clip_dim: Input CLIP embedding dimension (default: 768)
        sd_hidden_dim: SD text encoder hidden dimension (default: 768 for SD 1.5)
        n_tokens: Number of output tokens (default: 77)
        n_heads: Self-attention heads (default: 8)
        depth: Number of refiner layers (default: 2)
    """

    def __init__(
        self,
        clip_dim: int = 768,
        sd_hidden_dim: int = 768,
        n_tokens: int = 77,
        n_heads: int = 8,
        depth: int = 2,
    ):
        super().__init__()

        self.n_tokens = n_tokens
        self.sd_hidden_dim = sd_hidden_dim

        # Learnable query tokens
        self.query_tokens = nn.Parameter(
            torch.randn(n_tokens, sd_hidden_dim) * 0.02
        )

        # Project CLIP embedding to key/value for cross-attention
        self.clip_proj = nn.Linear(clip_dim, sd_hidden_dim)

        # Cross-attention: queries attend to projected CLIP embedding
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=sd_hidden_dim,
            num_heads=n_heads,
            batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(sd_hidden_dim)

        # Self-attention refinement layers
        self.refiner = nn.ModuleList()
        for _ in range(depth):
            self.refiner.append(
                nn.ModuleList([
                    nn.LayerNorm(sd_hidden_dim),
                    nn.MultiheadAttention(
                        embed_dim=sd_hidden_dim,
                        num_heads=n_heads,
                        batch_first=True,
                    ),
                    nn.LayerNorm(sd_hidden_dim),
                    nn.Sequential(
                        nn.Linear(sd_hidden_dim, sd_hidden_dim * 4),
                        nn.GELU(),
                        nn.Linear(sd_hidden_dim * 4, sd_hidden_dim),
                    ),
                ])
            )

        self.output_norm = nn.LayerNorm(sd_hidden_dim)

    def forward(self, clip_embedding: torch.Tensor) -> torch.Tensor:
        """Convert CLIP embedding to SD prompt embedding.

        Args:
            clip_embedding: (batch, clip_dim) normalized neural CLIP embedding

        Returns:
            (batch, n_tokens, sd_hidden_dim) SD-compatible prompt embedding
        """
        batch_size = clip_embedding.shape[0]

        # Project CLIP embedding
        clip_proj = self.clip_proj(clip_embedding).unsqueeze(1)  # (B, 1, sd_hidden_dim)

        # Expand query tokens for batch
        queries = self.query_tokens.unsqueeze(0).expand(batch_size, -1, -1)

        # Cross-attention: queries attend to CLIP embedding
        attended, _ = self.cross_attn(
            query=self.cross_norm(queries),
            key=clip_proj,
            value=clip_proj,
        )
        tokens = queries + attended

        # Self-attention refinement
        for norm1, self_attn, norm2, ffn in self.refiner:
            residual = tokens
            tokens_normed = norm1(tokens)
            attn_out, _ = self_attn(tokens_normed, tokens_normed, tokens_normed)
            tokens = residual + attn_out
            tokens = tokens + ffn(norm2(tokens))

        tokens = self.output_norm(tokens)

        return tokens
