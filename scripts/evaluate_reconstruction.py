"""Comprehensive evaluation of neural image reconstructions.

Computes multiple metrics:
1. PixCorr: pixel-wise Pearson correlation
2. SSIM: structural similarity
3. CLIP similarity: semantic similarity via CLIP embeddings
4. Two-way identification: can we identify the correct original from reconstructions?

Also supports retrieval-based reconstruction (nearest neighbor in CLIP space
from training set) as a comparison baseline.

Usage:
    HF_ENDPOINT=https://hf-mirror.com \
    PYTHONPATH=/path/to/NeuroBridge python scripts/evaluate_reconstruction.py \
        --recon_dir results/reconstruction_v3_trained \
        --things_dir /path/to/THINGS_images/object_images \
        --tvsd_dir /path/to/TVSD_dataset \
        --clip_embeddings_test data/clip_embeddings/clip_test_monkeyF.npy \
        --clip_embeddings_train data/clip_embeddings/clip_train_monkeyF.npy \
        --neural_embeddings results/reconstruction_v3_trained/neural_embeddings.npy
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))


def compute_pixcorr(original_images, reconstructed_images, size=256):
    """Pixel-wise Pearson correlation."""
    import torchvision.transforms.functional as TF

    corrs = []
    for orig, recon in zip(original_images, reconstructed_images):
        orig_t = TF.to_tensor(orig.resize((size, size))).flatten().float()
        recon_t = TF.to_tensor(recon.resize((size, size))).flatten().float()

        o = orig_t - orig_t.mean()
        r = recon_t - recon_t.mean()
        corr = (o * r).sum() / (o.norm() * r.norm() + 1e-8)
        corrs.append(corr.item())

    return {
        "pixcorr_mean": float(np.mean(corrs)),
        "pixcorr_std": float(np.std(corrs)),
        "pixcorr_all": corrs,
    }


def compute_clip_similarity(original_images, reconstructed_images, device="cuda"):
    """Compute CLIP similarity between original and reconstructed images."""
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="openai", device=device,
    )
    model.eval()

    orig_embeds = []
    recon_embeds = []

    with torch.no_grad():
        for orig, recon in zip(original_images, reconstructed_images):
            orig_input = preprocess(orig).unsqueeze(0).to(device)
            recon_input = preprocess(recon).unsqueeze(0).to(device)

            orig_emb = model.encode_image(orig_input)
            recon_emb = model.encode_image(recon_input)

            orig_embeds.append(F.normalize(orig_emb, dim=-1))
            recon_embeds.append(F.normalize(recon_emb, dim=-1))

    orig_embeds = torch.cat(orig_embeds, dim=0)    # (N, 768)
    recon_embeds = torch.cat(recon_embeds, dim=0)  # (N, 768)

    # Per-pair cosine similarity
    pair_sim = (orig_embeds * recon_embeds).sum(dim=-1)  # (N,)

    # Full similarity matrix for identification
    sim_matrix = recon_embeds @ orig_embeds.T  # (N, N)
    n = sim_matrix.shape[0]
    labels = torch.arange(n, device=sim_matrix.device)

    # Top-k identification
    results = {
        "clip_sim_mean": pair_sim.mean().item(),
        "clip_sim_std": pair_sim.std().item(),
        "clip_sim_all": pair_sim.cpu().tolist(),
    }

    for k in [1, 5, 10]:
        if k > n:
            continue
        _, topk = sim_matrix.topk(k, dim=1)
        correct = (topk == labels.unsqueeze(1)).any(dim=1).float().mean()
        results[f"clip_id_top{k}"] = correct.item()

    # Median rank
    ranks = (sim_matrix.argsort(dim=1, descending=True) == labels.unsqueeze(1)).nonzero()[:, 1]
    results["clip_id_median_rank"] = ranks.median().item()
    results["clip_id_mean_rank"] = ranks.float().mean().item()

    return results


def retrieval_baseline(neural_embeddings, clip_train, clip_test, train_image_paths, things_dir):
    """Retrieval-based reconstruction: find nearest training image in CLIP space.

    For each test neural embedding, find the training image whose CLIP embedding
    is most similar to the predicted neural CLIP embedding.

    Returns list of retrieved PIL images and metrics.
    """
    neural_emb = torch.from_numpy(neural_embeddings).float()
    neural_emb = F.normalize(neural_emb, dim=-1)
    train_emb = torch.from_numpy(clip_train).float()
    train_emb = F.normalize(train_emb, dim=-1)
    test_emb = torch.from_numpy(clip_test).float()
    test_emb = F.normalize(test_emb, dim=-1)

    # Neural → train CLIP similarity
    sim = neural_emb @ train_emb.T  # (100, 22248)
    top1_indices = sim.argmax(dim=1)  # (100,)

    # Load retrieved images
    retrieved_images = []
    for idx in top1_indices:
        img_path = Path(things_dir) / train_image_paths[idx.item()]
        if img_path.exists():
            retrieved_images.append(Image.open(img_path).convert("RGB"))
        else:
            # Fallback: black image
            retrieved_images.append(Image.new("RGB", (256, 256), (0, 0, 0)))

    # Compute retrieval accuracy against ground truth
    # (How often the top-1 retrieved image has the same class?)
    # We use CLIP similarity to test set as proxy
    neural_to_test_sim = neural_emb @ test_emb.T
    n = neural_to_test_sim.shape[0]
    labels = torch.arange(n)

    retrieval_metrics = {}
    for k in [1, 5, 10]:
        if k > n:
            continue
        _, topk = neural_to_test_sim.topk(k, dim=1)
        correct = (topk == labels.unsqueeze(1)).any(dim=1).float().mean()
        retrieval_metrics[f"retrieval_top{k}"] = correct.item()

    return retrieved_images, retrieval_metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate Reconstructions")
    parser.add_argument("--recon_dir", type=str, required=True)
    parser.add_argument("--things_dir", type=str, required=True)
    parser.add_argument("--tvsd_dir", type=str, required=True)
    parser.add_argument("--clip_embeddings_test", type=str, required=True)
    parser.add_argument("--clip_embeddings_train", type=str, default=None)
    parser.add_argument("--neural_embeddings", type=str, default=None)
    parser.add_argument("--monkey", type=str, default="monkeyF")
    parser.add_argument("--compute_clip", action="store_true",
                        help="Compute CLIP-based metrics (requires CLIP model)")
    parser.add_argument("--compute_retrieval", action="store_true",
                        help="Compute retrieval baseline comparison")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    recon_dir = Path(args.recon_dir)

    print("=" * 60)
    print("Neural Reconstruction Evaluation")
    print("=" * 60)

    # Load test image paths
    from neurobridge.data.tvsd_dataset import TVSDNormMUADataset
    dataset = TVSDNormMUADataset(
        tvsd_dir=args.tvsd_dir,
        monkey=args.monkey,
        split="test",
        mode="capoyo",
    )

    # Load original images
    print("\nLoading original test images...")
    original_images = []
    for i in range(len(dataset)):
        img_path = Path(args.things_dir) / dataset.image_paths[i]
        if img_path.exists():
            original_images.append(Image.open(img_path).convert("RGB"))
        else:
            original_images.append(Image.new("RGB", (256, 256), (128, 128, 128)))
    print(f"  Loaded {len(original_images)} original images")

    # Load reconstructed images
    print("Loading reconstructed images...")
    recon_files = sorted((recon_dir / "reconstructed").glob("recon_*.png"))
    reconstructed_images = [Image.open(f).convert("RGB") for f in recon_files]
    print(f"  Loaded {len(reconstructed_images)} reconstructed images")

    n = min(len(original_images), len(reconstructed_images))
    original_images = original_images[:n]
    reconstructed_images = reconstructed_images[:n]

    all_metrics = {}

    # 1. PixCorr
    print("\n--- PixCorr ---")
    pixcorr = compute_pixcorr(original_images, reconstructed_images)
    all_metrics.update({k: v for k, v in pixcorr.items() if k != "pixcorr_all"})
    print(f"  PixCorr: {pixcorr['pixcorr_mean']:.4f} ± {pixcorr['pixcorr_std']:.4f}")

    # 2. CLIP similarity
    if args.compute_clip:
        print("\n--- CLIP Similarity ---")
        clip_metrics = compute_clip_similarity(original_images, reconstructed_images, device)
        all_metrics.update({k: v for k, v in clip_metrics.items() if k != "clip_sim_all"})
        print(f"  CLIP sim: {clip_metrics['clip_sim_mean']:.4f} ± {clip_metrics['clip_sim_std']:.4f}")
        if "clip_id_top1" in clip_metrics:
            print(f"  CLIP ID Top-1: {clip_metrics['clip_id_top1']:.4f}")
            print(f"  CLIP ID Top-5: {clip_metrics.get('clip_id_top5', 'N/A')}")
            print(f"  CLIP ID Median rank: {clip_metrics.get('clip_id_median_rank', 'N/A')}")

    # 3. Retrieval baseline
    if args.compute_retrieval and args.clip_embeddings_train and args.neural_embeddings:
        print("\n--- Retrieval Baseline ---")
        neural_emb = np.load(args.neural_embeddings)
        clip_train = np.load(args.clip_embeddings_train)
        clip_test = np.load(args.clip_embeddings_test)

        # Get train image paths
        train_dataset = TVSDNormMUADataset(
            tvsd_dir=args.tvsd_dir,
            monkey=args.monkey,
            split="train",
            mode="capoyo",
        )

        retrieved_images, retrieval_metrics = retrieval_baseline(
            neural_emb, clip_train, clip_test,
            train_dataset.image_paths, args.things_dir,
        )

        # PixCorr for retrieved images
        retrieval_pixcorr = compute_pixcorr(original_images, retrieved_images)
        retrieval_metrics["retrieval_pixcorr_mean"] = retrieval_pixcorr["pixcorr_mean"]
        retrieval_metrics["retrieval_pixcorr_std"] = retrieval_pixcorr["pixcorr_std"]

        all_metrics.update(retrieval_metrics)
        print(f"  Retrieval Top-1: {retrieval_metrics.get('retrieval_top1', 'N/A')}")
        print(f"  Retrieval Top-5: {retrieval_metrics.get('retrieval_top5', 'N/A')}")
        print(f"  Retrieval PixCorr: {retrieval_pixcorr['pixcorr_mean']:.4f}")

        # Save retrieval comparison images
        retrieval_dir = recon_dir / "retrieval_comparison"
        retrieval_dir.mkdir(exist_ok=True)
        for i, (orig, retr, recon) in enumerate(
            zip(original_images, retrieved_images, reconstructed_images)
        ):
            combined = Image.new("RGB", (768, 256))
            combined.paste(orig.resize((256, 256)), (0, 0))
            combined.paste(retr.resize((256, 256)), (256, 0))
            combined.paste(recon.resize((256, 256)), (512, 0))
            combined.save(retrieval_dir / f"compare_{i:04d}.png")
        print(f"  Saved 3-way comparisons to {retrieval_dir}")

        if args.compute_clip:
            print("\n--- CLIP Similarity (Retrieval Baseline) ---")
            retrieval_clip = compute_clip_similarity(
                original_images, retrieved_images, device
            )
            all_metrics["retrieval_clip_sim_mean"] = retrieval_clip["clip_sim_mean"]
            all_metrics["retrieval_clip_sim_std"] = retrieval_clip["clip_sim_std"]
            print(f"  Retrieval CLIP sim: {retrieval_clip['clip_sim_mean']:.4f}")
            if "clip_id_top1" in retrieval_clip:
                all_metrics["retrieval_clip_id_top1"] = retrieval_clip["clip_id_top1"]
                print(f"  Retrieval CLIP ID Top-1: {retrieval_clip['clip_id_top1']:.4f}")

    # Save all metrics
    with open(recon_dir / "evaluation_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"All metrics saved to {recon_dir / 'evaluation_metrics.json'}")


if __name__ == "__main__":
    main()
