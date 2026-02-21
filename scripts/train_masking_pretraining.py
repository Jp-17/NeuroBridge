"""Masked MUA reconstruction pretraining for NeuroBridge.

Self-supervised pretraining: mask random electrodes, encode the remaining,
decode to reconstruct the masked values.

Usage:
    PYTHONPATH=/path/to/NeuroBridge python scripts/train_masking_pretraining.py \
        --tvsd_dir /path/to/TVSD_dataset \
        --monkey monkeyF \
        --epochs 200 \
        --mask_ratio 0.25
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, str(Path(__file__).parent.parent))

from neurobridge.data.tvsd_dataset import TVSDNormMUADataset
from neurobridge.pretraining.masking_pretraining import MaskingPretrainingModel
from scripts.train_clip_alignment import collate_fn


def train_epoch(model, optimizer, dataloader, device, scaler=None):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_all_loss = 0
    total_mask_ratio = 0
    n_batches = 0

    for batch in dataloader:
        _ = batch.pop("image_idx")  # not used for pretraining
        batch = {k: v.to(device) for k, v in batch.items()}

        optimizer.zero_grad()

        if scaler is not None:
            with torch.amp.autocast("cuda"):
                result = model(**batch)
                loss = result["loss"]
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            result = model(**batch)
            loss = result["loss"]
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += result["loss"].item()
        total_all_loss += result["all_loss"].item()
        total_mask_ratio += result["mask_ratio_actual"]
        n_batches += 1

    return {
        "masked_loss": total_loss / n_batches,
        "all_loss": total_all_loss / n_batches,
        "mask_ratio": total_mask_ratio / n_batches,
    }


@torch.no_grad()
def evaluate(model, dataloader, device):
    """Evaluate on validation set."""
    model.eval()
    total_loss = 0
    total_all_loss = 0
    n_batches = 0

    for batch in dataloader:
        _ = batch.pop("image_idx")
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.amp.autocast("cuda"):
            result = model(**batch)

        total_loss += result["loss"].item()
        total_all_loss += result["all_loss"].item()
        n_batches += 1

    return {
        "masked_loss": total_loss / n_batches,
        "all_loss": total_all_loss / n_batches,
    }


def main():
    parser = argparse.ArgumentParser(description="Masking Pretraining")
    parser.add_argument("--tvsd_dir", type=str, required=True)
    parser.add_argument("--monkey", type=str, default="monkeyF")
    parser.add_argument("--regions", type=str, nargs="*", default=None)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--mask_ratio", type=float, default=0.25)
    parser.add_argument("--encoder_dim", type=int, default=128)
    parser.add_argument("--encoder_depth", type=int, default=6)
    parser.add_argument("--num_latents", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="checkpoints/masking_pretrain_v1")
    parser.add_argument("--log_interval", type=int, default=5)
    parser.add_argument("--early_stopping", type=int, default=30)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("NeuroBridge Masking Pretraining")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Mask ratio: {args.mask_ratio}")

    # Dataset
    dataset = TVSDNormMUADataset(
        tvsd_dir=args.tvsd_dir,
        monkey=args.monkey,
        split="train",
        regions=args.regions,
        mode="capoyo",
    )
    stats = dataset.get_data_stats()
    print(f"Dataset: {stats['n_samples']} samples, {stats['n_electrodes']} electrodes")

    # Train/val split
    n_total = len(dataset)
    n_val = int(n_total * args.val_ratio)
    n_train = n_total - n_val
    train_dataset, val_dataset = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed)
    )
    print(f"Train: {n_train}, Val: {n_val}")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True, collate_fn=collate_fn, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True, collate_fn=collate_fn,
    )

    # Model
    model = MaskingPretrainingModel(
        encoder_dim=args.encoder_dim,
        encoder_depth=args.encoder_depth,
        num_latents=args.num_latents,
        mask_ratio=args.mask_ratio,
    ).to(device)

    # Initialize unit vocabularies
    unit_ids = dataset.get_unit_ids()
    model.encoder.unit_emb.initialize_vocab(unit_ids)
    model.decoder.query_unit_emb.initialize_vocab(unit_ids)

    n_params = model.count_parameters()
    print(f"Model parameters: {n_params:,}")
    print(f"  Encoder: {model.encoder.count_parameters():,}")
    print(f"  Decoder: {model.decoder.count_parameters():,}")

    # Optimizer
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
    with open(save_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Training loop
    best_val_loss = float("inf")
    patience_counter = 0
    log_lines = []

    print(f"\nStarting training for {args.epochs} epochs...")
    print("-" * 80)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_metrics = train_epoch(model, optimizer, train_loader, device, scaler)
        val_metrics = evaluate(model, val_loader, device)
        scheduler.step()

        dt = time.time() - t0

        log_line = (
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"Train masked: {train_metrics['masked_loss']:.6f} all: {train_metrics['all_loss']:.6f} "
            f"mask%: {train_metrics['mask_ratio']:.3f} | "
            f"Val masked: {val_metrics['masked_loss']:.6f} all: {val_metrics['all_loss']:.6f} | "
            f"lr: {scheduler.get_last_lr()[0]:.2e} | {dt:.1f}s"
        )
        log_lines.append(log_line)

        if epoch % args.log_interval == 0 or epoch == 1:
            print(log_line)

        # Save best model
        if val_metrics["masked_loss"] < best_val_loss:
            best_val_loss = val_metrics["masked_loss"]
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "encoder_state_dict": model.encoder.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": best_val_loss,
                "config": config,
            }, save_dir / "best_model.pt")
        else:
            patience_counter += 1

        if epoch % 50 == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "encoder_state_dict": model.encoder.state_dict(),
                "val_loss": val_metrics["masked_loss"],
                "config": config,
            }, save_dir / f"checkpoint_epoch{epoch}.pt")

        if args.early_stopping > 0 and patience_counter >= args.early_stopping:
            print(f"Early stopping at epoch {epoch}")
            break

    with open(save_dir / "training_log.txt", "w") as f:
        f.write("\n".join(log_lines))

    print("-" * 80)
    print(f"Training complete. Best val masked loss: {best_val_loss:.6f}")
    print(f"Checkpoints saved to {save_dir}")


if __name__ == "__main__":
    main()
