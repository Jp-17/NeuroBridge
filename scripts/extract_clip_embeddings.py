"""Extract CLIP embeddings for THINGS images used in TVSD.

Pre-computes CLIP (ViT-L-14) embeddings for all train and test images,
saves as .npy files for use during CLIP alignment training.

Usage:
    PYTHONPATH=/path/to/NeuroBridge python scripts/extract_clip_embeddings.py \
        --things_dir /path/to/THINGS_images/object_images \
        --tvsd_dir /path/to/TVSD_dataset \
        --monkey monkeyF \
        --output_dir /path/to/embeddings \
        --batch_size 64
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import h5py
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_image_paths(tvsd_dir, monkey):
    """Load image paths from things_imgs.mat."""
    things_path = Path(tvsd_dir) / monkey / "_logs" / "things_imgs.mat"

    train_paths = []
    test_paths = []

    with h5py.File(things_path, "r") as f:
        # Train images
        train_refs = f["train_imgs"]["things_path"]
        for i in range(train_refs.shape[0]):
            ref = train_refs[i, 0]
            obj = f[ref]
            chars = obj[:]
            path = "".join(chr(c) for c in chars.flat if c > 0)
            train_paths.append(path.replace("\\", "/"))

        # Test images
        test_refs = f["test_imgs"]["things_path"]
        for i in range(test_refs.shape[0]):
            ref = test_refs[i, 0]
            obj = f[ref]
            chars = obj[:]
            path = "".join(chr(c) for c in chars.flat if c > 0)
            test_paths.append(path.replace("\\", "/"))

    return train_paths, test_paths


def extract_embeddings(model, preprocess, image_dir, image_paths, batch_size, device):
    """Extract CLIP embeddings for a list of images."""
    all_embeddings = []
    n = len(image_paths)
    n_missing = 0

    for i in range(0, n, batch_size):
        batch_paths = image_paths[i : i + batch_size]
        images = []
        valid_mask = []

        for p in batch_paths:
            full_path = Path(image_dir) / p
            if full_path.exists():
                try:
                    img = Image.open(full_path).convert("RGB")
                    images.append(preprocess(img))
                    valid_mask.append(True)
                except Exception as e:
                    print(f"  Warning: failed to load {p}: {e}")
                    valid_mask.append(False)
                    n_missing += 1
            else:
                valid_mask.append(False)
                n_missing += 1

        if images:
            batch_tensor = torch.stack(images).to(device)
            with torch.no_grad():
                features = model.encode_image(batch_tensor)
                features = features / features.norm(dim=-1, keepdim=True)
                features = features.cpu().numpy()

            # Insert zeros for missing images
            batch_embeddings = np.zeros((len(batch_paths), features.shape[1]), dtype=np.float32)
            j = 0
            for k, valid in enumerate(valid_mask):
                if valid:
                    batch_embeddings[k] = features[j]
                    j += 1
            all_embeddings.append(batch_embeddings)
        else:
            # All missing in this batch
            embed_dim = 768  # ViT-L-14 default
            all_embeddings.append(np.zeros((len(batch_paths), embed_dim), dtype=np.float32))

        processed = min(i + batch_size, n)
        if processed % (batch_size * 10) == 0 or processed == n:
            print(f"  Processed {processed}/{n} images ({n_missing} missing)")

    all_embeddings = np.concatenate(all_embeddings, axis=0)
    print(f"  Total: {n} images, {n_missing} missing, {n - n_missing} extracted")
    return all_embeddings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--things_dir", type=str, required=True,
                        help="Path to THINGS images directory (containing class folders)")
    parser.add_argument("--tvsd_dir", type=str, required=True,
                        help="Path to TVSD_dataset directory")
    parser.add_argument("--monkey", type=str, default="monkeyF")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save embeddings .npy files")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--model_name", type=str, default="ViT-L-14")
    parser.add_argument("--pretrained", type=str, default="openai")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("CLIP Embedding Extraction for THINGS/TVSD")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"THINGS dir: {args.things_dir}")
    print(f"Monkey: {args.monkey}")

    # Load image paths
    print("\nLoading image paths from things_imgs.mat...")
    train_paths, test_paths = load_image_paths(args.tvsd_dir, args.monkey)
    print(f"  Train images: {len(train_paths)}")
    print(f"  Test images: {len(test_paths)}")

    # Verify some images exist
    things_dir = Path(args.things_dir)
    n_found = sum(1 for p in train_paths[:100] if (things_dir / p).exists())
    print(f"  First 100 train images found: {n_found}/100")

    if n_found == 0:
        print("ERROR: No images found! Check --things_dir path.")
        print(f"  Expected structure: {things_dir}/aardvark/aardvark_01b.jpg")
        sys.exit(1)

    # Load CLIP model
    print(f"\nLoading CLIP model: {args.model_name} ({args.pretrained})...")
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model_name, pretrained=args.pretrained
    )
    model = model.to(device)
    model.eval()
    print(f"  Model loaded. Embed dim: {model.visual.output_dim}")

    # Extract train embeddings
    print(f"\nExtracting train embeddings ({len(train_paths)} images)...")
    train_embeddings = extract_embeddings(
        model, preprocess, args.things_dir, train_paths, args.batch_size, device
    )
    train_path = output_dir / f"clip_train_{args.monkey}.npy"
    np.save(train_path, train_embeddings)
    print(f"  Saved: {train_path} shape={train_embeddings.shape}")

    # Extract test embeddings
    print(f"\nExtracting test embeddings ({len(test_paths)} images)...")
    test_embeddings = extract_embeddings(
        model, preprocess, args.things_dir, test_paths, args.batch_size, device
    )
    test_path = output_dir / f"clip_test_{args.monkey}.npy"
    np.save(test_path, test_embeddings)
    print(f"  Saved: {test_path} shape={test_embeddings.shape}")

    print("\nDone!")


if __name__ == "__main__":
    main()
