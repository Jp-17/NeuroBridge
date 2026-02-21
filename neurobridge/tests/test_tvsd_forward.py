"""Smoke test: verify TVSD normMUA data flows through NeuroBridge encoder.

This script:
1. Loads TVSD normMUA dataset
2. Creates a NeuroBridge encoder model
3. Runs a forward pass
4. Verifies output shapes

Usage:
    PYTHONPATH=/root/autodl-tmp/NeuroBridge \
    conda run -n poyo python neurobridge/tests/test_tvsd_forward.py
"""

import sys
import torch
import numpy as np

sys.path.insert(0, "/root/autodl-tmp/NeuroBridge")

from neurobridge.data.tvsd_dataset import TVSDNormMUADataset
from neurobridge.models.neurobridge_encoder import NeuroBridgeEncoder
from torch_brain.data import pad8, track_mask8


def test_dataset_loading():
    """Test TVSD dataset loads correctly."""
    print("=" * 60)
    print("TEST 1: Dataset Loading")
    print("=" * 60)

    tvsd_dir = "/root/autodl-tmp/TVSD_dataset"

    # Test raw mode
    dataset = TVSDNormMUADataset(tvsd_dir, monkey="monkeyF", split="train", mode="raw")
    stats = dataset.get_data_stats()
    print(f"Dataset stats: {stats}")
    print(f"  N samples: {len(dataset)}")
    print(f"  N electrodes: {dataset.n_active_electrodes}")

    sample = dataset[0]
    print(f"  Sample keys: {list(sample.keys())}")
    print(f"  MUA shape: {sample['mua'].shape}")
    print(f"  Image class: {sample['image_class']}")
    print(f"  Image path: {sample['image_path']}")
    print(f"  MUA range: [{sample['mua'].min():.3f}, {sample['mua'].max():.3f}]")

    # Test with region filtering
    dataset_v1 = TVSDNormMUADataset(
        tvsd_dir, monkey="monkeyF", split="train", mode="raw", regions=["V1"]
    )
    print(f"\n  V1 only: {dataset_v1.n_active_electrodes} electrodes")

    dataset_it = TVSDNormMUADataset(
        tvsd_dir, monkey="monkeyF", split="train", mode="raw", regions=["IT"]
    )
    print(f"  IT only: {dataset_it.n_active_electrodes} electrodes")

    dataset_v4 = TVSDNormMUADataset(
        tvsd_dir, monkey="monkeyF", split="train", mode="raw", regions=["V4"]
    )
    print(f"  V4 only: {dataset_v4.n_active_electrodes} electrodes")

    print("  PASSED")
    return dataset


def test_capoyo_mode():
    """Test CaPOYO tokenization mode."""
    print("\n" + "=" * 60)
    print("TEST 2: CaPOYO Tokenization Mode")
    print("=" * 60)

    tvsd_dir = "/root/autodl-tmp/TVSD_dataset"
    dataset = TVSDNormMUADataset(
        tvsd_dir, monkey="monkeyF", split="train", mode="capoyo"
    )

    sample = dataset[0]
    print(f"  Sample keys: {list(sample.keys())}")
    print(f"  input_values shape: {sample['input_values'].shape}")
    print(f"  input_timestamps shape: {sample['input_timestamps'].shape}")
    print(f"  input_unit_index shape: {sample['input_unit_index'].shape}")
    print(f"  latent_index shape: {sample['latent_index'].shape}")
    print(f"  latent_timestamps shape: {sample['latent_timestamps'].shape}")
    print(f"  n_electrodes: {sample['n_electrodes']}")
    print(f"  Image class: {sample['image_class']}")
    print("  PASSED")
    return dataset


def test_forward_pass():
    """Test full forward pass through NeuroBridge encoder."""
    print("\n" + "=" * 60)
    print("TEST 3: Forward Pass")
    print("=" * 60)

    tvsd_dir = "/root/autodl-tmp/TVSD_dataset"
    dataset = TVSDNormMUADataset(
        tvsd_dir, monkey="monkeyF", split="train", mode="capoyo"
    )

    # Create model
    model = NeuroBridgeEncoder(
        dim=128,
        depth=6,
        dim_head=64,
        cross_heads=2,
        self_heads=4,
        num_latents=8,
    )

    # Initialize vocabularies
    unit_ids = dataset.get_unit_ids()
    model.unit_emb.initialize_vocab(unit_ids)

    n_params = model.count_parameters()
    print(f"  Model parameters: {n_params:,}")

    # Create a mini-batch (2 samples)
    samples = [dataset[0], dataset[1]]

    # Manual batching with padding
    batch_size = len(samples)
    max_n = max(s["input_values"].shape[0] for s in samples)

    # Pad to same length and batch
    input_values_list = []
    input_timestamps_list = []
    input_unit_index_list = []
    input_mask_list = []

    for s in samples:
        n = s["input_values"].shape[0]
        pad_n = max_n - n
        # Pad
        iv = torch.cat([s["input_values"], torch.zeros(pad_n, 1)], dim=0)
        it = torch.cat(
            [s["input_timestamps"], torch.zeros(pad_n, dtype=torch.float64)], dim=0
        )
        iu = torch.cat(
            [s["input_unit_index"], torch.zeros(pad_n, dtype=torch.int64)], dim=0
        )
        mask = torch.cat(
            [torch.ones(n, dtype=torch.bool), torch.zeros(pad_n, dtype=torch.bool)],
            dim=0,
        )

        input_values_list.append(iv)
        input_timestamps_list.append(it)
        input_unit_index_list.append(iu)
        input_mask_list.append(mask)

    input_values = torch.stack(input_values_list)
    input_timestamps = torch.stack(input_timestamps_list)
    input_unit_index = torch.stack(input_unit_index_list)
    input_mask = torch.stack(input_mask_list)

    # Latent tokens (same for all samples)
    latent_index = samples[0]["latent_index"].unsqueeze(0).expand(batch_size, -1)
    latent_timestamps = (
        samples[0]["latent_timestamps"].unsqueeze(0).expand(batch_size, -1)
    )

    print(f"  Batch shapes:")
    print(f"    input_values: {input_values.shape}")
    print(f"    input_timestamps: {input_timestamps.shape}")
    print(f"    input_unit_index: {input_unit_index.shape}")
    print(f"    input_mask: {input_mask.shape}")
    print(f"    latent_index: {latent_index.shape}")
    print(f"    latent_timestamps: {latent_timestamps.shape}")

    # Forward pass
    model.eval()
    with torch.no_grad():
        latents = model(
            input_unit_index=input_unit_index,
            input_timestamps=input_timestamps,
            input_values=input_values,
            input_mask=input_mask,
            latent_index=latent_index,
            latent_timestamps=latent_timestamps,
        )

    print(f"\n  Output latent shape: {latents.shape}")
    print(f"  Expected: ({batch_size}, {model.num_latents}, {model.dim})")
    assert latents.shape == (
        batch_size,
        model.num_latents,
        model.dim,
    ), f"Shape mismatch!"

    # Test pooled representation
    pooled = model.get_latent_representation(latents)
    print(f"  Pooled shape: {pooled.shape}")
    assert pooled.shape == (batch_size, model.dim), "Pooled shape mismatch!"

    print(f"  Latent mean: {latents.mean():.6f}")
    print(f"  Latent std: {latents.std():.6f}")
    print("  PASSED")

    return model, dataset


def test_gpu_forward():
    """Test forward pass on GPU."""
    print("\n" + "=" * 60)
    print("TEST 4: GPU Forward Pass")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("  SKIPPED (no GPU)")
        return

    tvsd_dir = "/root/autodl-tmp/TVSD_dataset"
    dataset = TVSDNormMUADataset(
        tvsd_dir, monkey="monkeyF", split="train", mode="capoyo"
    )

    model = NeuroBridgeEncoder(
        dim=128,
        depth=6,
        dim_head=64,
        cross_heads=2,
        self_heads=4,
        num_latents=8,
    )
    model.unit_emb.initialize_vocab(dataset.get_unit_ids())
    model = model.cuda()

    # Single sample forward
    sample = dataset[0]
    input_values = sample["input_values"].unsqueeze(0).cuda()
    input_timestamps = sample["input_timestamps"].unsqueeze(0).cuda()
    input_unit_index = sample["input_unit_index"].unsqueeze(0).cuda()
    latent_index = sample["latent_index"].unsqueeze(0).cuda()
    latent_timestamps = sample["latent_timestamps"].unsqueeze(0).cuda()

    model.eval()
    with torch.no_grad():
        latents = model(
            input_unit_index=input_unit_index,
            input_timestamps=input_timestamps,
            input_values=input_values,
            latent_index=latent_index,
            latent_timestamps=latent_timestamps,
        )

    print(f"  GPU output shape: {latents.shape}")
    print(f"  GPU memory: {torch.cuda.memory_allocated()/1024**2:.1f} MB")
    print("  PASSED")


def test_batch_forward():
    """Test larger batch forward pass on GPU with AMP."""
    print("\n" + "=" * 60)
    print("TEST 5: Batch Forward (GPU + AMP)")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("  SKIPPED (no GPU)")
        return

    tvsd_dir = "/root/autodl-tmp/TVSD_dataset"
    dataset = TVSDNormMUADataset(
        tvsd_dir, monkey="monkeyF", split="train", mode="capoyo"
    )

    model = NeuroBridgeEncoder(
        dim=128,
        depth=6,
        dim_head=64,
        cross_heads=2,
        self_heads=4,
        num_latents=8,
    )
    model.unit_emb.initialize_vocab(dataset.get_unit_ids())
    model = model.cuda()

    batch_size = 32
    samples = [dataset[i] for i in range(batch_size)]

    # Stack batch
    input_values = torch.stack([s["input_values"] for s in samples]).cuda()
    input_timestamps = torch.stack([s["input_timestamps"] for s in samples]).cuda()
    input_unit_index = torch.stack([s["input_unit_index"] for s in samples]).cuda()
    latent_index = samples[0]["latent_index"].unsqueeze(0).expand(batch_size, -1).cuda()
    latent_timestamps = (
        samples[0]["latent_timestamps"].unsqueeze(0).expand(batch_size, -1).cuda()
    )

    # Forward with AMP
    model.eval()
    with torch.no_grad(), torch.cuda.amp.autocast():
        latents = model(
            input_unit_index=input_unit_index,
            input_timestamps=input_timestamps,
            input_values=input_values,
            latent_index=latent_index,
            latent_timestamps=latent_timestamps,
        )

    print(f"  Batch size: {batch_size}")
    print(f"  Output shape: {latents.shape}")
    print(f"  GPU memory: {torch.cuda.memory_allocated()/1024**2:.1f} MB")
    print(f"  Max GPU memory: {torch.cuda.max_memory_allocated()/1024**2:.1f} MB")
    print("  PASSED")


if __name__ == "__main__":
    print("NeuroBridge Phase 1a: TVSD Forward Pass Verification")
    print("=" * 60)

    test_dataset_loading()
    test_capoyo_mode()
    test_forward_pass()
    test_gpu_forward()
    test_batch_forward()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
