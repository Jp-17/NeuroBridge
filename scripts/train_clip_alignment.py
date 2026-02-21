"""End-to-end CLIP alignment training for NeuroBridge.

Trains the neural encoder + readout + projector to align MUA representations
with CLIP image embeddings using InfoNCE contrastive loss.

Training pipeline:
  TVSD normMUA → NeuroBridgeEncoder → NeuralReadout → NeuralProjector → InfoNCE ← CLIP embeddings

Usage:
    PYTHONPATH=/path/to/NeuroBridge python scripts/train_clip_alignment.py \
        --tvsd_dir /path/to/TVSD_dataset \
        --clip_embeddings /path/to/clip_embeddings.npy \
        --monkey monkeyF \
        --epochs 100 \
        --batch_size 256 \
        --lr 3e-4

If --clip_embeddings is not provided, random embeddings are used (for pipeline verification).
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, str(Path(__file__).parent.parent))

from neurobridge.data.tvsd_dataset import TVSDNormMUADataset
from neurobridge.models.neurobridge_encoder import NeuroBridgeEncoder
from neurobridge.alignment.readout import NeuralReadout
from neurobridge.alignment.projector import NeuralProjector
from neurobridge.alignment.infonce import InfoNCELoss


class NeuroBridgeAlignmentModel(nn.Module):
    """Full pipeline: Encoder + Readout + Projector for CLIP alignment."""

    def __init__(
        self,
        encoder_dim: int = 128,
        encoder_depth: int = 6,
        encoder_dim_head: int = 64,
        encoder_cross_heads: int = 2,
        encoder_self_heads: int = 4,
        num_latents: int = 8,
        readout_queries: int = 8,
        readout_heads: int = 2,
        projector_hidden: int = 512,
        clip_dim: int = 768,
        pool_method: str = "mean",
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
        self.readout = NeuralReadout(
            dim=encoder_dim,
            n_queries=readout_queries,
            n_heads=readout_heads,
            dim_head=encoder_dim_head,
        )
        self.projector = NeuralProjector(
            input_dim=encoder_dim,
            hidden_dim=projector_hidden,
            output_dim=clip_dim,
            n_queries=readout_queries,
            pool_method=pool_method,
        )

    def forward(self, input_unit_index, input_timestamps, input_values,
                input_mask, latent_index, latent_timestamps):
        """Full forward: MUA → encoder → readout → projector → CLIP-space embedding."""
        # Encode
        latents = self.encoder(
            input_unit_index=input_unit_index,
            input_timestamps=input_timestamps,
            input_values=input_values,
            input_mask=input_mask,
            latent_index=latent_index,
            latent_timestamps=latent_timestamps,
        )
        # Read out
        readout_tokens = self.readout(latents)
        # Project to CLIP space
        neural_embeds = self.projector(readout_tokens)
        return neural_embeds

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def collate_fn(batch):
    """Custom collate function for TVSD capoyo samples."""
    # All samples have same number of electrodes (1024), so simple stacking works
    input_values = torch.stack([s["input_values"] for s in batch])
    input_timestamps = torch.stack([s["input_timestamps"] for s in batch])
    input_unit_index = torch.stack([s["input_unit_index"] for s in batch])
    latent_index = batch[0]["latent_index"].unsqueeze(0).expand(len(batch), -1).clone()
    latent_timestamps = batch[0]["latent_timestamps"].unsqueeze(0).expand(len(batch), -1).clone()

    return {
        "input_values": input_values,
        "input_timestamps": input_timestamps,
        "input_unit_index": input_unit_index,
        "input_mask": torch.ones(input_values.shape[0], input_values.shape[1], dtype=torch.bool),
        "latent_index": latent_index,
        "latent_timestamps": latent_timestamps,
        "image_idx": torch.tensor([s["idx"] for s in batch], dtype=torch.long),
    }


def augment_neural_data(batch, electrode_dropout=0.1, noise_std=0.1):
    """Apply augmentation to neural MUA data during training.

    Args:
        batch: dict with input_values, input_mask, etc.
        electrode_dropout: probability of zeroing out each electrode
        noise_std: std of additive Gaussian noise on MUA values
    """
    input_values = batch["input_values"]  # (B, N, 1)
    input_mask = batch["input_mask"]      # (B, N)

    # 1. Electrode dropout: randomly mask out electrodes
    if electrode_dropout > 0:
        drop_mask = torch.rand(input_values.shape[0], input_values.shape[1],
                               device=input_values.device) > electrode_dropout
        input_values = input_values * drop_mask.unsqueeze(-1).float()
        input_mask = input_mask & drop_mask

    # 2. Gaussian noise on MUA values
    if noise_std > 0:
        noise = torch.randn_like(input_values) * noise_std
        input_values = input_values + noise

    batch["input_values"] = input_values
    batch["input_mask"] = input_mask
    return batch


def train_epoch(model, criterion, optimizer, dataloader, clip_embeddings,
                device, epoch, scaler=None, electrode_dropout=0.0, noise_std=0.0):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_acc = 0
    total_top5 = 0
    n_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        # Get CLIP embeddings for this batch
        indices = batch.pop("image_idx")
        clip_emb = clip_embeddings[indices].to(device)
        clip_emb = F.normalize(clip_emb, dim=-1)

        # Move batch to device
        batch = {k: v.to(device) for k, v in batch.items()}

        # Apply augmentation during training
        if electrode_dropout > 0 or noise_std > 0:
            batch = augment_neural_data(batch, electrode_dropout, noise_std)

        optimizer.zero_grad()

        if scaler is not None:
            with torch.amp.autocast("cuda"):
                neural_emb = model(**batch)
                result = criterion(neural_emb, clip_emb)
                loss = result["loss"]
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            neural_emb = model(**batch)
            result = criterion(neural_emb, clip_emb)
            loss = result["loss"]
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item()
        total_acc += result["accuracy"]
        total_top5 += result["top5_accuracy"]
        n_batches += 1

    return {
        "loss": total_loss / n_batches,
        "accuracy": total_acc / n_batches,
        "top5_accuracy": total_top5 / n_batches,
    }


@torch.no_grad()
def evaluate(model, criterion, dataloader, clip_embeddings, device):
    """Evaluate on validation set."""
    model.eval()
    total_loss = 0
    total_acc = 0
    total_top5 = 0
    n_batches = 0

    for batch in dataloader:
        indices = batch.pop("image_idx")
        clip_emb = clip_embeddings[indices].to(device)
        clip_emb = F.normalize(clip_emb, dim=-1)

        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.amp.autocast("cuda"):
            neural_emb = model(**batch)
            result = criterion(neural_emb, clip_emb)

        total_loss += result["loss"].item()
        total_acc += result["accuracy"]
        total_top5 += result["top5_accuracy"]
        n_batches += 1

    return {
        "loss": total_loss / n_batches,
        "accuracy": total_acc / n_batches,
        "top5_accuracy": total_top5 / n_batches,
    }


def main():
    parser = argparse.ArgumentParser(description="NeuroBridge CLIP Alignment Training")
    parser.add_argument("--tvsd_dir", type=str, required=True,
                        help="Path to TVSD_dataset directory")
    parser.add_argument("--clip_embeddings", type=str, default=None,
                        help="Path to pre-computed CLIP embeddings .npy file")
    parser.add_argument("--monkey", type=str, default="monkeyF",
                        choices=["monkeyF", "monkeyN"])
    parser.add_argument("--regions", type=str, nargs="*", default=None,
                        help="Brain regions to use (V1, V4, IT)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--encoder_dim", type=int, default=128)
    parser.add_argument("--encoder_depth", type=int, default=6)
    parser.add_argument("--num_latents", type=int, default=8)
    parser.add_argument("--clip_dim", type=int, default=768)
    parser.add_argument("--projector_hidden", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--log_interval", type=int, default=5)
    parser.add_argument("--electrode_dropout", type=float, default=0.0,
                        help="Electrode dropout rate for augmentation")
    parser.add_argument("--noise_std", type=float, default=0.0,
                        help="Gaussian noise std for MUA augmentation")
    parser.add_argument("--early_stopping", type=int, default=0,
                        help="Early stopping patience (0=disabled)")
    parser.add_argument("--pretrained_encoder", type=str, default=None,
                        help="Path to pretrained encoder checkpoint (masking pretraining)")
    parser.add_argument("--freeze_encoder_epochs", type=int, default=0,
                        help="Freeze encoder for first N epochs (warmup readout/projector)")
    args = parser.parse_args()

    # Setup
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("NeuroBridge CLIP Alignment Training")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Monkey: {args.monkey}")
    print(f"Regions: {args.regions or 'all'}")

    # Load dataset
    dataset = TVSDNormMUADataset(
        tvsd_dir=args.tvsd_dir,
        monkey=args.monkey,
        split="train",
        regions=args.regions,
        mode="capoyo",
    )
    stats = dataset.get_data_stats()
    print(f"Dataset: {stats['n_samples']} samples, {stats['n_electrodes']} electrodes, "
          f"{stats['n_classes']} classes")

    # Train/val split
    n_total = len(dataset)
    n_val = int(n_total * args.val_ratio)
    n_train = n_total - n_val
    train_dataset, val_dataset = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed)
    )
    print(f"Train: {n_train}, Val: {n_val}")

    # CLIP embeddings
    if args.clip_embeddings and os.path.exists(args.clip_embeddings):
        print(f"Loading CLIP embeddings from {args.clip_embeddings}")
        clip_embeddings = torch.from_numpy(np.load(args.clip_embeddings)).float()
    else:
        print("WARNING: Using random CLIP embeddings (no real CLIP embeddings provided)")
        print("  This is for pipeline verification only. Results will be meaningless.")
        clip_embeddings = torch.randn(n_total, args.clip_dim)
        clip_embeddings = F.normalize(clip_embeddings, dim=-1)

    # DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True, collate_fn=collate_fn, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True, collate_fn=collate_fn,
    )

    # Model
    model = NeuroBridgeAlignmentModel(
        encoder_dim=args.encoder_dim,
        encoder_depth=args.encoder_depth,
        num_latents=args.num_latents,
        clip_dim=args.clip_dim,
        projector_hidden=args.projector_hidden,
    ).to(device)

    # Initialize vocabularies
    model.encoder.unit_emb.initialize_vocab(dataset.get_unit_ids())

    # Load pretrained encoder if provided
    if args.pretrained_encoder:
        print(f"Loading pretrained encoder from {args.pretrained_encoder}")
        pretrain_ckpt = torch.load(args.pretrained_encoder, map_location=device, weights_only=False)
        if "encoder_state_dict" in pretrain_ckpt:
            model.encoder.load_state_dict(pretrain_ckpt["encoder_state_dict"])
            print(f"  Loaded encoder from epoch {pretrain_ckpt.get('epoch', '?')}")
        else:
            # Try loading from full model state dict
            full_state = pretrain_ckpt["model_state_dict"]
            encoder_state = {k.replace("encoder.", ""): v for k, v in full_state.items()
                            if k.startswith("encoder.")}
            model.encoder.load_state_dict(encoder_state)
            print(f"  Loaded encoder from full model checkpoint")

    n_params = model.count_parameters()
    print(f"Model parameters: {n_params:,}")

    # Loss and optimizer
    criterion = InfoNCELoss(temperature=0.07, learnable_temperature=True).to(device)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(criterion.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )
    scaler = torch.amp.GradScaler("cuda")

    # Save config
    config = vars(args)
    config["n_params"] = n_params
    config["n_train"] = n_train
    config["n_val"] = n_val
    with open(save_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Training loop
    best_val_top5 = 0.0
    log_lines = []
    patience_counter = 0

    print(f"\nStarting training for {args.epochs} epochs...")
    if args.electrode_dropout > 0 or args.noise_std > 0:
        print(f"  Augmentation: electrode_dropout={args.electrode_dropout}, noise_std={args.noise_std}")
    if args.early_stopping > 0:
        print(f"  Early stopping patience: {args.early_stopping}")
    if args.pretrained_encoder:
        print(f"  Pretrained encoder: {args.pretrained_encoder}")
    if args.freeze_encoder_epochs > 0:
        print(f"  Freezing encoder for first {args.freeze_encoder_epochs} epochs")
        for param in model.encoder.parameters():
            param.requires_grad = False
    print("-" * 80)

    for epoch in range(1, args.epochs + 1):
        # Unfreeze encoder after warmup
        if args.freeze_encoder_epochs > 0 and epoch == args.freeze_encoder_epochs + 1:
            print(f"  [Epoch {epoch}] Unfreezing encoder")
            for param in model.encoder.parameters():
                param.requires_grad = True

        t0 = time.time()

        train_metrics = train_epoch(
            model, criterion, optimizer, train_loader, clip_embeddings,
            device, epoch, scaler,
            electrode_dropout=args.electrode_dropout,
            noise_std=args.noise_std,
        )
        val_metrics = evaluate(model, criterion, val_loader, clip_embeddings, device)
        scheduler.step()

        dt = time.time() - t0

        log_line = (
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"Train loss: {train_metrics['loss']:.4f} acc: {train_metrics['accuracy']:.4f} "
            f"top5: {train_metrics['top5_accuracy']:.4f} | "
            f"Val loss: {val_metrics['loss']:.4f} acc: {val_metrics['accuracy']:.4f} "
            f"top5: {val_metrics['top5_accuracy']:.4f} | "
            f"lr: {scheduler.get_last_lr()[0]:.2e} | "
            f"{dt:.1f}s"
        )
        log_lines.append(log_line)

        if epoch % args.log_interval == 0 or epoch == 1:
            print(log_line)

        # Save best model
        if val_metrics["top5_accuracy"] > best_val_top5:
            best_val_top5 = val_metrics["top5_accuracy"]
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_top5": best_val_top5,
                "config": config,
            }, save_dir / "best_model.pt")
        else:
            patience_counter += 1

        # Save latest
        if epoch % 10 == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_top5": val_metrics["top5_accuracy"],
                "config": config,
            }, save_dir / f"checkpoint_epoch{epoch}.pt")

        # Early stopping
        if args.early_stopping > 0 and patience_counter >= args.early_stopping:
            print(f"Early stopping at epoch {epoch} (patience={args.early_stopping})")
            break

    # Save training log
    with open(save_dir / "training_log.txt", "w") as f:
        f.write("\n".join(log_lines))

    print("-" * 80)
    print(f"Training complete. Best val top-5: {best_val_top5:.4f}")
    print(f"Checkpoints saved to {save_dir}")


if __name__ == "__main__":
    main()
