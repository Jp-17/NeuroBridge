"""CLIP model wrapper for NeuroBridge.

Handles loading CLIP models and extracting image/text embeddings.
Supports both OpenAI CLIP (via open_clip) and HuggingFace transformers.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Optional, List, Union
from PIL import Image


class CLIPWrapper(nn.Module):
    """Wrapper for CLIP model to extract image embeddings.

    Args:
        model_name: CLIP model name (default: 'ViT-L-14')
        pretrained: Pretrained weights (default: 'openai')
        device: Device to run on
        cache_dir: Directory to cache model weights
    """

    def __init__(
        self,
        model_name: str = "ViT-L-14",
        pretrained: str = "openai",
        device: str = "cuda",
        cache_dir: Optional[str] = None,
    ):
        super().__init__()
        self.device = device
        self.model_name = model_name

        import open_clip

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, cache_dir=cache_dir
        )
        self.model = self.model.to(device)
        self.model.eval()

        # Freeze all parameters
        for p in self.model.parameters():
            p.requires_grad = False

        # Get embedding dimension
        self.embed_dim = self.model.visual.output_dim

    @torch.no_grad()
    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """Encode preprocessed image tensor to CLIP embedding.

        Args:
            image: Preprocessed image tensor (B, 3, H, W)

        Returns:
            Normalized CLIP embedding (B, embed_dim)
        """
        features = self.model.encode_image(image.to(self.device))
        features = features / features.norm(dim=-1, keepdim=True)
        return features

    @torch.no_grad()
    def encode_image_from_path(self, image_path: str) -> torch.Tensor:
        """Load and encode a single image from file path.

        Args:
            image_path: Path to image file

        Returns:
            Normalized CLIP embedding (1, embed_dim)
        """
        img = Image.open(image_path).convert("RGB")
        img_tensor = self.preprocess(img).unsqueeze(0)
        return self.encode_image(img_tensor)

    @torch.no_grad()
    def batch_encode_images(
        self,
        image_dir: str,
        image_paths: List[str],
        batch_size: int = 64,
        save_path: Optional[str] = None,
    ) -> np.ndarray:
        """Extract CLIP embeddings for a batch of images.

        Args:
            image_dir: Root directory for THINGS images
            image_paths: List of relative paths (e.g., "aardvark/aardvark_01b.jpg")
            batch_size: Batch size for extraction
            save_path: If provided, save embeddings to .npy file

        Returns:
            Array of CLIP embeddings (N, embed_dim)
        """
        all_embeddings = []
        n = len(image_paths)

        for i in range(0, n, batch_size):
            batch_paths = image_paths[i : i + batch_size]
            images = []
            for p in batch_paths:
                full_path = Path(image_dir) / p
                img = Image.open(full_path).convert("RGB")
                images.append(self.preprocess(img))

            batch_tensor = torch.stack(images)
            embeddings = self.encode_image(batch_tensor)
            all_embeddings.append(embeddings.cpu().numpy())

            if (i // batch_size) % 10 == 0:
                print(f"  Processed {min(i + batch_size, n)}/{n} images")

        all_embeddings = np.concatenate(all_embeddings, axis=0)

        if save_path:
            np.save(save_path, all_embeddings)
            print(f"  Saved embeddings to {save_path}: shape={all_embeddings.shape}")

        return all_embeddings

    def get_embed_dim(self) -> int:
        """Return the CLIP embedding dimension."""
        return self.embed_dim
