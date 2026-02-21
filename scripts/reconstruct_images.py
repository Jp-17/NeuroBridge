"""End-to-end neural image reconstruction pipeline.

Takes a trained CLIP alignment model + DiffusionAdapter to reconstruct
images from TVSD normMUA neural data.

Pipeline:
  normMUA → NeuroBridgeEncoder → NeuralReadout → NeuralProjector → CLIP embedding
    → DiffusionAdapter → SD prompt embedding → StableDiffusion → reconstructed image

Usage:
    HF_ENDPOINT=https://hf-mirror.com \
    PYTHONPATH=/path/to/NeuroBridge python scripts/reconstruct_images.py \
        --alignment_checkpoint checkpoints/clip_alignment_v1/best_model.pt \
        --tvsd_dir /path/to/TVSD_dataset \
        --clip_embeddings data/clip_embeddings/clip_test_monkeyF.npy \
        --things_dir /path/to/THINGS_images/object_images \
        --monkey monkeyF \
        --split test \
        --output_dir results/reconstruction_v1
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from neurobridge.data.tvsd_dataset import TVSDNormMUADataset
from neurobridge.generation.diffusion_adapter import DiffusionAdapter, StableDiffusionWrapper
from scripts.train_clip_alignment import NeuroBridgeAlignmentModel, collate_fn


@torch.no_grad()
def extract_neural_embeddings(model, dataset, device, batch_size=64):
    """Extract neural CLIP embeddings for all samples."""
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

    return torch.cat(all_embeddings, dim=0), torch.cat(all_indices, dim=0)


def compute_metrics(original_images, reconstructed_images):
    """Compute reconstruction metrics between original and reconstructed images.

    Returns dict with PixCorr, SSIM, and basic statistics.
    """
    import torchvision.transforms.functional as TF

    metrics = {}
    pix_corrs = []
    ssim_vals = []

    for orig, recon in zip(original_images, reconstructed_images):
        # Convert to tensors and resize to same dimensions
        if isinstance(orig, Image.Image):
            orig_t = TF.to_tensor(orig.resize((256, 256)))
        else:
            orig_t = orig
        if isinstance(recon, Image.Image):
            recon_t = TF.to_tensor(recon.resize((256, 256)))
        else:
            recon_t = recon

        # PixCorr: pixel-wise Pearson correlation
        o = orig_t.flatten().float()
        r = recon_t.flatten().float()
        o_centered = o - o.mean()
        r_centered = r - r.mean()
        pix_corr = (o_centered * r_centered).sum() / (
            o_centered.norm() * r_centered.norm() + 1e-8
        )
        pix_corrs.append(pix_corr.item())

    metrics["pixcorr_mean"] = float(np.mean(pix_corrs))
    metrics["pixcorr_std"] = float(np.std(pix_corrs))

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Neural Image Reconstruction")
    parser.add_argument("--alignment_checkpoint", type=str, required=True)
    parser.add_argument("--tvsd_dir", type=str, required=True)
    parser.add_argument("--clip_embeddings", type=str, required=True,
                        help="Pre-computed CLIP embeddings for retrieval comparison")
    parser.add_argument("--things_dir", type=str, default=None,
                        help="Path to THINGS images for side-by-side comparison")
    parser.add_argument("--monkey", type=str, default="monkeyF")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--output_dir", type=str, default="results/reconstruction_v1")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_samples", type=int, default=None,
                        help="Number of samples to reconstruct (None=all)")
    parser.add_argument("--sd_model", type=str, default="stabilityai/stable-diffusion-2-1")
    parser.add_argument("--sd_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_sd", action="store_true",
                        help="Skip SD generation, only compute retrieval metrics")
    parser.add_argument("--adapter_checkpoint", type=str, default=None,
                        help="Path to trained DiffusionAdapter checkpoint")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("NeuroBridge Neural Image Reconstruction")
    print("=" * 60)

    # --- Step 1: Load alignment model ---
    print(f"\n[1/5] Loading alignment model: {args.alignment_checkpoint}")
    ckpt = torch.load(args.alignment_checkpoint, map_location=device, weights_only=False)
    config = ckpt["config"]
    print(f"  Epoch: {ckpt['epoch']}, Val top5: {ckpt.get('val_top5', 'N/A')}")

    model = NeuroBridgeAlignmentModel(
        encoder_dim=config.get("encoder_dim", 128),
        encoder_depth=config.get("encoder_depth", 6),
        num_latents=config.get("num_latents", 8),
        clip_dim=config.get("clip_dim", 768),
        projector_hidden=config.get("projector_hidden", 512),
    ).to(device)

    dataset = TVSDNormMUADataset(
        tvsd_dir=args.tvsd_dir,
        monkey=args.monkey,
        split=args.split,
        mode="capoyo",
    )
    model.encoder.unit_emb.initialize_vocab(dataset.get_unit_ids())
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"  Dataset: {len(dataset)} samples")

    # --- Step 2: Extract neural embeddings ---
    print(f"\n[2/5] Extracting neural embeddings...")
    neural_emb, indices = extract_neural_embeddings(
        model, dataset, device, args.batch_size
    )
    neural_emb = F.normalize(neural_emb, dim=-1)
    print(f"  Neural embeddings: {neural_emb.shape}")

    # --- Step 3: Retrieval evaluation ---
    print(f"\n[3/5] Computing retrieval metrics...")
    clip_embeddings = torch.from_numpy(np.load(args.clip_embeddings)).float()
    clip_emb = F.normalize(clip_embeddings[indices], dim=-1)

    # Cosine similarity matrix
    sim = neural_emb @ clip_emb.T
    n = sim.shape[0]
    labels = torch.arange(n)

    retrieval_results = {}
    for k in [1, 5, 10]:
        if k > n:
            continue
        _, topk = sim.topk(k, dim=1)
        correct = (topk == labels.unsqueeze(1)).any(dim=1).float().mean()
        retrieval_results[f"n2i_top{k}"] = correct.item()
        print(f"  Neural→Image Top-{k}: {correct.item():.4f}")

    # Median rank
    ranks = (sim.argsort(dim=1, descending=True) == labels.unsqueeze(1)).nonzero()[:, 1]
    retrieval_results["median_rank"] = ranks.median().item()
    retrieval_results["mean_rank"] = ranks.float().mean().item()
    print(f"  Median rank: {retrieval_results['median_rank']:.0f}/{n}")
    print(f"  Mean rank: {retrieval_results['mean_rank']:.1f}/{n}")

    # Positive/negative similarity
    diag_sim = sim.diag()
    mask = ~torch.eye(n, dtype=bool)
    retrieval_results["mean_pos_sim"] = diag_sim.mean().item()
    retrieval_results["mean_neg_sim"] = sim[mask].mean().item()
    print(f"  Positive sim: {retrieval_results['mean_pos_sim']:.4f}")
    print(f"  Negative sim: {retrieval_results['mean_neg_sim']:.4f}")

    # Save retrieval results
    with open(output_dir / "retrieval_metrics.json", "w") as f:
        json.dump(retrieval_results, f, indent=2)

    if args.skip_sd:
        print("\n[4/5] Skipping SD generation (--skip_sd)")
        print("[5/5] Skipping image saving")
        print(f"\nResults saved to {output_dir}")
        return

    # --- Step 4: Generate images with SD ---
    n_gen = args.n_samples or n
    print(f"\n[4/5] Generating {n_gen} images with Stable Diffusion...")

    # Load DiffusionAdapter
    adapter = DiffusionAdapter(
        clip_dim=config.get("clip_dim", 768),
        sd_hidden_dim=1024,  # SD 2.1 uses 1024
        n_tokens=77,
    ).to(device)

    if args.adapter_checkpoint:
        adapter_ckpt = torch.load(args.adapter_checkpoint, map_location=device)
        adapter.load_state_dict(adapter_ckpt["model_state_dict"])
    adapter.eval()

    # Load SD
    sd_wrapper = StableDiffusionWrapper(
        model_id=args.sd_model,
        device=str(device),
    )

    reconstructed_images = []
    original_images = []

    for i in range(n_gen):
        emb = neural_emb[i:i+1].to(device)

        # Expand through adapter
        with torch.amp.autocast("cuda"):
            prompt_embeds = adapter(emb)

        # Generate with SD
        prompt_embeds_fp16 = prompt_embeds.to(torch.float16)
        negative_embeds = torch.zeros_like(prompt_embeds_fp16)

        gen = torch.Generator(device=device).manual_seed(args.seed + i)
        images = sd_wrapper.pipe(
            prompt_embeds=prompt_embeds_fp16,
            negative_prompt_embeds=negative_embeds,
            num_inference_steps=args.sd_steps,
            guidance_scale=args.guidance_scale,
            generator=gen,
        ).images

        reconstructed_images.append(images[0])

        # Load original image if available
        if args.things_dir:
            img_path = Path(args.things_dir) / dataset.image_paths[indices[i]]
            if img_path.exists():
                original_images.append(Image.open(img_path).convert("RGB"))

        if (i + 1) % 10 == 0 or i == 0:
            print(f"  Generated {i + 1}/{n_gen}")

    # --- Step 5: Save results ---
    print(f"\n[5/5] Saving results...")

    # Save individual images
    recon_dir = output_dir / "reconstructed"
    recon_dir.mkdir(exist_ok=True)
    for i, img in enumerate(reconstructed_images):
        img.save(recon_dir / f"recon_{i:04d}.png")

    # Save side-by-side comparisons if originals available
    if original_images:
        compare_dir = output_dir / "comparison"
        compare_dir.mkdir(exist_ok=True)
        for i, (orig, recon) in enumerate(zip(original_images, reconstructed_images)):
            # Resize to same dimensions
            orig_resized = orig.resize((256, 256))
            recon_resized = recon.resize((256, 256))

            # Side-by-side
            combined = Image.new("RGB", (512, 256))
            combined.paste(orig_resized, (0, 0))
            combined.paste(recon_resized, (256, 0))
            combined.save(compare_dir / f"compare_{i:04d}.png")

        # Compute pixel-level metrics
        recon_metrics = compute_metrics(original_images, reconstructed_images)
        retrieval_results.update(recon_metrics)
        print(f"  PixCorr: {recon_metrics['pixcorr_mean']:.4f} ± {recon_metrics['pixcorr_std']:.4f}")

    # Save all metrics
    with open(output_dir / "all_metrics.json", "w") as f:
        json.dump(retrieval_results, f, indent=2)

    # Save neural embeddings for further analysis
    np.save(output_dir / "neural_embeddings.npy", neural_emb.numpy())

    print(f"\nReconstruction complete! Results saved to {output_dir}")
    print(f"  {len(reconstructed_images)} images reconstructed")
    if original_images:
        print(f"  {len(original_images)} comparisons saved")


if __name__ == "__main__":
    main()
