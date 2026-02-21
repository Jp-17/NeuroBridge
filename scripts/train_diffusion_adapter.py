"""Train DiffusionAdapter to expand CLIP embeddings into SD prompt embeddings.

The adapter learns to map a single 768-dim CLIP image embedding to a
(77, 768) sequence of prompt tokens that Stable Diffusion can use to
reconstruct the original image.

Training uses MSE loss between:
  - SD text encoder output for a CLIP-derived prompt
  - Adapter output from the CLIP image embedding

Since we don't have text captions, we use an alternative approach:
  1. Feed CLIP image embeddings through the adapter
  2. Generate images with SD using the adapter output
  3. Compute CLIP similarity between generated images and CLIP target embeddings
  4. Backprop through the adapter (not SD) using CLIP loss

Actually, the standard approach (MindEye, BrainDiffuser) trains the adapter
with a simpler proxy objective:
  - Use the SD text encoder to get reference prompt embeddings from text
  - Train adapter to match these embeddings

But since we have no captions, we use a self-supervised objective:
  - Train the adapter so that SD(adapter(clip_emb)) ≈ original_image
  - Use CLIP similarity + MSE in CLIP space as the loss

For efficiency, we use a diffusion prior approach (like DALL-E 2):
  - Train the adapter to produce prompt embeddings that, when used with SD,
    generate images whose CLIP embeddings match the input CLIP embeddings.

Simplified training approach:
  - Stage 1: Train adapter with MSE loss against reference SD prompt embeddings
             from CLIP text encoder (using simple category-name captions)
  - Stage 2: Fine-tune with CLIP-guided generation loss (optional, expensive)

Usage:
    HF_ENDPOINT=https://hf-mirror.com \
    PYTHONPATH=/path/to/NeuroBridge python scripts/train_diffusion_adapter.py \
        --clip_embeddings data/clip_embeddings/clip_train_monkeyF.npy \
        --tvsd_dir /path/to/TVSD_dataset \
        --output_dir checkpoints/diffusion_adapter_v1
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

sys.path.insert(0, str(Path(__file__).parent.parent))

from neurobridge.generation.diffusion_adapter import DiffusionAdapter


class CLIPEmbeddingDataset(Dataset):
    """Simple dataset of CLIP embeddings for adapter training."""

    def __init__(self, embeddings: np.ndarray, labels: np.ndarray = None):
        self.embeddings = torch.from_numpy(embeddings).float()
        self.embeddings = F.normalize(self.embeddings, dim=-1)
        self.labels = labels

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        item = {"clip_emb": self.embeddings[idx]}
        if self.labels is not None:
            item["label"] = self.labels[idx]
        return item


class AdapterReconstructionLoss(nn.Module):
    """Loss for training DiffusionAdapter.

    Combines multiple objectives:
    1. Token diversity loss: encourages different tokens to capture different aspects
    2. Embedding preservation: first token should preserve the CLIP embedding
    3. Contrastive loss: adapter outputs for different images should be distinguishable

    Args:
        clip_dim: CLIP embedding dimension
        n_tokens: Number of output tokens
        diversity_weight: Weight for token diversity loss
        preservation_weight: Weight for embedding preservation loss
        contrastive_weight: Weight for contrastive loss
    """

    def __init__(
        self,
        clip_dim: int = 768,
        n_tokens: int = 77,
        diversity_weight: float = 0.1,
        preservation_weight: float = 1.0,
        contrastive_weight: float = 0.5,
    ):
        super().__init__()
        self.diversity_weight = diversity_weight
        self.preservation_weight = preservation_weight
        self.contrastive_weight = contrastive_weight

    def forward(self, adapter_output, clip_input):
        """Compute training loss.

        Args:
            adapter_output: (batch, n_tokens, sd_hidden_dim) from adapter
            clip_input: (batch, clip_dim) input CLIP embeddings

        Returns:
            Dict with loss components
        """
        batch_size = adapter_output.shape[0]
        n_tokens = adapter_output.shape[1]
        losses = {}

        # 1. Embedding preservation: mean of tokens should approximate input embedding
        mean_token = adapter_output.mean(dim=1)  # (batch, dim)
        mean_token_norm = F.normalize(mean_token, dim=-1)
        clip_norm = F.normalize(clip_input, dim=-1)

        # Cosine similarity loss (want to maximize similarity)
        cos_sim = (mean_token_norm * clip_norm).sum(dim=-1)  # (batch,)
        preservation_loss = 1.0 - cos_sim.mean()
        losses["preservation"] = preservation_loss

        # 2. Token diversity: tokens should not all be the same
        # Compute pairwise cosine similarity between tokens
        tokens_norm = F.normalize(adapter_output, dim=-1)  # (batch, n_tokens, dim)
        # Mean pairwise similarity (should be low for diverse tokens)
        token_sim = torch.bmm(tokens_norm, tokens_norm.transpose(1, 2))  # (batch, n, n)
        # Exclude diagonal
        mask = ~torch.eye(n_tokens, device=token_sim.device, dtype=torch.bool).unsqueeze(0)
        mean_pairwise_sim = token_sim[mask.expand(batch_size, -1, -1)].mean()
        diversity_loss = mean_pairwise_sim  # want to minimize
        losses["diversity"] = diversity_loss

        # 3. Contrastive loss: different inputs should produce distinguishable outputs
        # Pool adapter outputs to single vectors, then InfoNCE
        pooled = F.normalize(mean_token, dim=-1)  # (batch, dim)
        sim_matrix = pooled @ clip_norm.T  # (batch, batch)
        sim_matrix = sim_matrix / 0.1  # temperature
        labels = torch.arange(batch_size, device=sim_matrix.device)
        contrastive_loss = 0.5 * (
            F.cross_entropy(sim_matrix, labels) +
            F.cross_entropy(sim_matrix.T, labels)
        )
        losses["contrastive"] = contrastive_loss

        # 4. L2 regularization on token norms (prevent explosion)
        token_norms = adapter_output.norm(dim=-1)  # (batch, n_tokens)
        norm_loss = ((token_norms - 1.0) ** 2).mean()
        losses["norm_reg"] = norm_loss

        # Total loss
        total = (
            self.preservation_weight * preservation_loss +
            self.diversity_weight * diversity_loss +
            self.contrastive_weight * contrastive_loss +
            0.01 * norm_loss
        )
        losses["total"] = total

        # Metrics
        with torch.no_grad():
            # Top-1 contrastive accuracy
            preds = sim_matrix.argmax(dim=1)
            accuracy = (preds == labels).float().mean().item()
            losses["accuracy"] = accuracy
            losses["mean_cos_sim"] = cos_sim.mean().item()
            losses["mean_token_norm"] = token_norms.mean().item()

        return losses


def train_epoch(model, criterion, optimizer, dataloader, device, scaler=None):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_acc = 0
    total_cos = 0
    n_batches = 0

    for batch in dataloader:
        clip_emb = batch["clip_emb"].to(device)

        optimizer.zero_grad()

        if scaler is not None:
            with torch.amp.autocast("cuda"):
                output = model(clip_emb)
                result = criterion(output, clip_emb)
                loss = result["total"]
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            output = model(clip_emb)
            result = criterion(output, clip_emb)
            loss = result["total"]
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item()
        total_acc += result["accuracy"]
        total_cos += result["mean_cos_sim"]
        n_batches += 1

    return {
        "loss": total_loss / n_batches,
        "accuracy": total_acc / n_batches,
        "cos_sim": total_cos / n_batches,
    }


@torch.no_grad()
def evaluate(model, criterion, dataloader, device):
    """Evaluate on validation set."""
    model.eval()
    total_loss = 0
    total_acc = 0
    total_cos = 0
    n_batches = 0

    for batch in dataloader:
        clip_emb = batch["clip_emb"].to(device)

        with torch.amp.autocast("cuda"):
            output = model(clip_emb)
            result = criterion(output, clip_emb)

        total_loss += result["total"].item()
        total_acc += result["accuracy"]
        total_cos += result["mean_cos_sim"]
        n_batches += 1

    return {
        "loss": total_loss / n_batches,
        "accuracy": total_acc / n_batches,
        "cos_sim": total_cos / n_batches,
    }


def main():
    parser = argparse.ArgumentParser(description="Train DiffusionAdapter")
    parser.add_argument("--clip_embeddings", type=str, required=True,
                        help="Path to CLIP embeddings .npy file")
    parser.add_argument("--output_dir", type=str, default="checkpoints/diffusion_adapter_v1")
    parser.add_argument("--clip_dim", type=int, default=768)
    parser.add_argument("--sd_hidden_dim", type=int, default=768)
    parser.add_argument("--n_tokens", type=int, default=77)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--early_stopping", type=int, default=30,
                        help="Early stopping patience (0=disabled)")
    # Loss weights
    parser.add_argument("--preservation_weight", type=float, default=1.0)
    parser.add_argument("--diversity_weight", type=float, default=0.1)
    parser.add_argument("--contrastive_weight", type=float, default=0.5)
    args = parser.parse_args()

    # Setup
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("DiffusionAdapter Training")
    print("=" * 60)
    print(f"Device: {device}")

    # Load CLIP embeddings
    print(f"\nLoading CLIP embeddings from {args.clip_embeddings}")
    clip_data = np.load(args.clip_embeddings)
    print(f"  Shape: {clip_data.shape}")

    dataset = CLIPEmbeddingDataset(clip_data)

    # Train/val split
    n_total = len(dataset)
    n_val = int(n_total * args.val_ratio)
    n_train = n_total - n_val
    train_dataset, val_dataset = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed)
    )
    print(f"  Train: {n_train}, Val: {n_val}")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True,
    )

    # Model
    model = DiffusionAdapter(
        clip_dim=args.clip_dim,
        sd_hidden_dim=args.sd_hidden_dim,
        n_tokens=args.n_tokens,
        n_heads=args.n_heads,
        depth=args.depth,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters: {n_params:,}")

    # Loss and optimizer
    criterion = AdapterReconstructionLoss(
        clip_dim=args.clip_dim,
        n_tokens=args.n_tokens,
        diversity_weight=args.diversity_weight,
        preservation_weight=args.preservation_weight,
        contrastive_weight=args.contrastive_weight,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01,
    )
    scaler = torch.amp.GradScaler("cuda")

    # Save config
    config = vars(args)
    config["n_params"] = n_params
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Training loop
    best_val_loss = float("inf")
    best_val_cos = 0.0
    patience_counter = 0
    log_lines = []

    print(f"\nStarting training for {args.epochs} epochs...")
    print("-" * 80)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_metrics = train_epoch(model, criterion, optimizer, train_loader, device, scaler)
        val_metrics = evaluate(model, criterion, val_loader, device)
        scheduler.step()

        dt = time.time() - t0

        log_line = (
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"Train loss: {train_metrics['loss']:.4f} acc: {train_metrics['accuracy']:.4f} "
            f"cos: {train_metrics['cos_sim']:.4f} | "
            f"Val loss: {val_metrics['loss']:.4f} acc: {val_metrics['accuracy']:.4f} "
            f"cos: {val_metrics['cos_sim']:.4f} | "
            f"lr: {scheduler.get_last_lr()[0]:.2e} | {dt:.1f}s"
        )
        log_lines.append(log_line)

        if epoch % args.log_interval == 0 or epoch == 1:
            print(log_line)

        # Save best model (based on val cos similarity)
        if val_metrics["cos_sim"] > best_val_cos:
            best_val_cos = val_metrics["cos_sim"]
            best_val_loss = val_metrics["loss"]
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_cos_sim": best_val_cos,
                "val_loss": best_val_loss,
                "config": config,
            }, output_dir / "best_model.pt")
        else:
            patience_counter += 1

        # Save periodic checkpoint
        if epoch % 50 == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_cos_sim": val_metrics["cos_sim"],
                "config": config,
            }, output_dir / f"checkpoint_epoch{epoch}.pt")

        # Early stopping
        if args.early_stopping > 0 and patience_counter >= args.early_stopping:
            print(f"Early stopping at epoch {epoch} (patience={args.early_stopping})")
            break

    # Save training log
    with open(output_dir / "training_log.txt", "w") as f:
        f.write("\n".join(log_lines))

    print("-" * 80)
    print(f"Training complete. Best val cosine sim: {best_val_cos:.4f}")
    print(f"Checkpoints saved to {output_dir}")


if __name__ == "__main__":
    main()
