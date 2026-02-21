"""Evaluate CLIP alignment quality using retrieval metrics.

Computes:
- Top-1/5/10 retrieval accuracy (neural → image, image → neural)
- CLIP similarity distributions
- Per-class retrieval accuracy
- Confusion analysis

Usage:
    PYTHONPATH=/path/to/NeuroBridge python scripts/evaluate_alignment.py \
        --checkpoint /path/to/best_model.pt \
        --tvsd_dir /path/to/TVSD_dataset \
        --clip_embeddings /path/to/clip_test_monkeyF.npy \
        --monkey monkeyF \
        --split test
"""

import argparse
import json
import sys
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from neurobridge.data.tvsd_dataset import TVSDNormMUADataset
from scripts.train_clip_alignment import NeuroBridgeAlignmentModel, collate_fn


@torch.no_grad()
def extract_neural_embeddings(model, dataset, device, batch_size=128):
    """Extract neural embeddings for all samples in dataset."""
    from torch.utils.data import DataLoader

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, collate_fn=collate_fn,
    )

    all_embeddings = []
    all_indices = []

    model.eval()
    for batch in loader:
        indices = batch.pop("image_idx")
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.amp.autocast("cuda"):
            neural_emb = model(**batch)

        all_embeddings.append(neural_emb.cpu())
        all_indices.append(indices)

    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_indices = torch.cat(all_indices, dim=0)
    return all_embeddings, all_indices


def compute_retrieval_metrics(neural_emb, clip_emb, ks=[1, 5, 10, 50]):
    """Compute retrieval metrics between neural and CLIP embeddings."""
    # Normalize
    neural_emb = F.normalize(neural_emb, dim=-1)
    clip_emb = F.normalize(clip_emb, dim=-1)

    # Cosine similarity matrix
    sim = neural_emb @ clip_emb.T  # (N_neural, N_clip)

    n = sim.shape[0]
    labels = torch.arange(n)

    results = {}

    # Neural → Image retrieval
    for k in ks:
        if k > n:
            continue
        _, topk = sim.topk(k, dim=1)
        correct = (topk == labels.unsqueeze(1)).any(dim=1).float().mean()
        results[f"n2i_top{k}"] = correct.item()

    # Image → Neural retrieval
    for k in ks:
        if k > n:
            continue
        _, topk = sim.T.topk(k, dim=1)
        correct = (topk == labels.unsqueeze(1)).any(dim=1).float().mean()
        results[f"i2n_top{k}"] = correct.item()

    # Mean retrieval rank
    ranks_n2i = (sim.argsort(dim=1, descending=True) == labels.unsqueeze(1)).nonzero()[:, 1]
    ranks_i2n = (sim.T.argsort(dim=1, descending=True) == labels.unsqueeze(1)).nonzero()[:, 1]
    results["n2i_median_rank"] = ranks_n2i.median().item()
    results["i2n_median_rank"] = ranks_i2n.median().item()
    results["n2i_mean_rank"] = ranks_n2i.float().mean().item()
    results["i2n_mean_rank"] = ranks_i2n.float().mean().item()

    # Diagonal similarity statistics
    diag_sim = sim.diag()
    results["mean_positive_sim"] = diag_sim.mean().item()
    results["std_positive_sim"] = diag_sim.std().item()

    # Off-diagonal similarity
    mask = ~torch.eye(n, dtype=bool)
    off_diag = sim[mask]
    results["mean_negative_sim"] = off_diag.mean().item()
    results["std_negative_sim"] = off_diag.std().item()

    return results


def compute_per_class_accuracy(neural_emb, clip_emb, image_classes, k=5):
    """Compute per-class retrieval accuracy."""
    neural_emb = F.normalize(neural_emb, dim=-1)
    clip_emb = F.normalize(clip_emb, dim=-1)
    sim = neural_emb @ clip_emb.T

    n = sim.shape[0]
    labels = torch.arange(n)
    _, topk = sim.topk(k, dim=1)
    correct = (topk == labels.unsqueeze(1)).any(dim=1)

    class_acc = {}
    for i, cls in enumerate(image_classes):
        if cls not in class_acc:
            class_acc[cls] = {"correct": 0, "total": 0}
        class_acc[cls]["total"] += 1
        if correct[i]:
            class_acc[cls]["correct"] += 1

    for cls in class_acc:
        class_acc[cls]["accuracy"] = (
            class_acc[cls]["correct"] / class_acc[cls]["total"]
        )

    return class_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--tvsd_dir", type=str, required=True)
    parser.add_argument("--clip_embeddings", type=str, required=True)
    parser.add_argument("--monkey", type=str, default="monkeyF")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("NeuroBridge CLIP Alignment Evaluation")
    print("=" * 60)

    # Load checkpoint
    print(f"\nLoading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = ckpt["config"]
    print(f"  Epoch: {ckpt['epoch']}, Val top5: {ckpt.get('val_top5', 'N/A')}")

    # Load dataset
    dataset = TVSDNormMUADataset(
        tvsd_dir=args.tvsd_dir,
        monkey=args.monkey,
        split=args.split,
        mode="capoyo",
    )
    print(f"  Dataset: {len(dataset)} samples")

    # Load CLIP embeddings
    clip_embeddings = torch.from_numpy(np.load(args.clip_embeddings)).float()
    print(f"  CLIP embeddings: {clip_embeddings.shape}")

    # Build model
    model = NeuroBridgeAlignmentModel(
        encoder_dim=config.get("encoder_dim", 128),
        encoder_depth=config.get("encoder_depth", 6),
        num_latents=config.get("num_latents", 8),
        clip_dim=config.get("clip_dim", 768),
        projector_hidden=config.get("projector_hidden", 512),
    ).to(device)
    model.encoder.unit_emb.initialize_vocab(dataset.get_unit_ids())
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Extract neural embeddings
    print("\nExtracting neural embeddings...")
    neural_emb, indices = extract_neural_embeddings(
        model, dataset, device, args.batch_size
    )
    clip_emb = clip_embeddings[indices]

    # Compute retrieval metrics
    print("\nComputing retrieval metrics...")
    metrics = compute_retrieval_metrics(neural_emb, clip_emb)

    print("\n" + "=" * 60)
    print("RETRIEVAL RESULTS")
    print("=" * 60)
    print(f"  Neural → Image:")
    for k in [1, 5, 10, 50]:
        key = f"n2i_top{k}"
        if key in metrics:
            print(f"    Top-{k}: {metrics[key]:.4f}")
    print(f"    Median rank: {metrics['n2i_median_rank']:.0f}")

    print(f"\n  Image → Neural:")
    for k in [1, 5, 10, 50]:
        key = f"i2n_top{k}"
        if key in metrics:
            print(f"    Top-{k}: {metrics[key]:.4f}")
    print(f"    Median rank: {metrics['i2n_median_rank']:.0f}")

    print(f"\n  Similarity:")
    print(f"    Positive (mean±std): {metrics['mean_positive_sim']:.4f} ± {metrics['std_positive_sim']:.4f}")
    print(f"    Negative (mean±std): {metrics['mean_negative_sim']:.4f} ± {metrics['std_negative_sim']:.4f}")

    # Save results
    if args.output:
        with open(args.output, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
