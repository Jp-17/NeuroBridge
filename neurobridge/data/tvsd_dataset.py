"""TVSD normMUA dataset adapter for NeuroBridge.

Loads THINGS_normMUA.mat and things_imgs.mat from the TVSD dataset
and provides a PyTorch Dataset that outputs CaPOYO-compatible data.

TVSD normMUA is time-averaged (2D): each trial = 1024 electrode values (no time dim).
We treat each electrode as a single "event" at time=0, matching CaPOYO's tokenization.

Electrode-to-brain-region mapping (from norm_MUA.m):
  monkeyF: ch 1-512 = V1, ch 513-832 = IT, ch 833-1024 = V4
  monkeyN: ch 1-512 = V1, ch 513-768 = V4, ch 769-1024 = IT
"""

import numpy as np
import h5py
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional, Dict, List, Tuple


# Brain region assignments per monkey
REGION_MAP = {
    "monkeyF": {
        "V1": (0, 512),    # ch 1-512
        "IT": (512, 832),  # ch 513-832
        "V4": (832, 1024), # ch 833-1024
    },
    "monkeyN": {
        "V1": (0, 512),    # ch 1-512
        "V4": (512, 768),  # ch 513-768
        "IT": (768, 1024), # ch 769-1024
    },
}


def _deref_cell_array(h5file, dataset, idx):
    """Dereference a MATLAB cell array element stored as HDF5 object reference."""
    ref = dataset[idx, 0]
    obj = h5file[ref]
    chars = obj[:]
    return "".join(chr(c) for c in chars.flat if c > 0)


class TVSDNormMUADataset(Dataset):
    """TVSD normalized MUA dataset for CaPOYO-style neural encoding.

    Each sample is one image trial: 1024 electrode MUA values + image metadata.

    The dataset can output data in two modes:
    1. 'raw': returns (mua_vector, image_info_dict) — for simple MLP models
    2. 'capoyo': returns a dict compatible with CaPOYO model.tokenize() format

    Args:
        tvsd_dir: Path to TVSD_dataset directory (e.g., .../TVSD_dataset)
        monkey: 'monkeyF' or 'monkeyN'
        split: 'train' or 'test'
        regions: List of brain regions to include, e.g., ['V1', 'V4', 'IT'].
                 None means all regions.
        mode: 'raw' or 'capoyo'
        snr_threshold: Minimum SNR_max to include an electrode. None = no filtering.
    """

    def __init__(
        self,
        tvsd_dir: str,
        monkey: str = "monkeyF",
        split: str = "train",
        regions: Optional[List[str]] = None,
        mode: str = "raw",
        snr_threshold: Optional[float] = None,
    ):
        super().__init__()
        self.tvsd_dir = Path(tvsd_dir)
        self.monkey = monkey
        self.split = split
        self.regions = regions
        self.mode = mode
        self.snr_threshold = snr_threshold

        # Load normMUA data
        normMUA_path = self.tvsd_dir / monkey / "THINGS_normMUA.mat"
        assert normMUA_path.exists(), f"normMUA not found: {normMUA_path}"

        with h5py.File(normMUA_path, "r") as f:
            if split == "train":
                # h5py shape: (22248, 1024) — stimuli × electrodes
                # (MATLAB stores electrodes × stimuli, h5py auto-transposes)
                self.mua_data = np.array(f["train_MUA"]).astype(np.float32)
            elif split == "test":
                # h5py shape: (100, 1024)
                self.mua_data = np.array(f["test_MUA"]).astype(np.float32)
            elif split == "test_reps":
                # h5py shape: (30, 100, 1024) — reps × stimuli × electrodes
                self.mua_data = np.array(f["test_MUA_reps"]).astype(np.float32)
            else:
                raise ValueError(f"Unknown split: {split}")

            # Load quality metrics
            self.snr_max = np.array(f["SNR_max"]).flatten().astype(np.float32)

        # Load image metadata
        things_path = self.tvsd_dir / monkey / "_logs" / "things_imgs.mat"
        assert things_path.exists(), f"things_imgs not found: {things_path}"
        self._load_image_metadata(things_path)

        # Determine electrode indices for selected regions
        self.n_electrodes = 1024
        self.electrode_mask = self._compute_electrode_mask()
        self.active_electrode_indices = np.where(self.electrode_mask)[0]
        self.n_active_electrodes = len(self.active_electrode_indices)

        # Generate unit IDs for active electrodes
        self.unit_ids = np.array(
            [f"{monkey}_ch{i:04d}" for i in self.active_electrode_indices]
        )

        # Compute per-electrode region labels (1=V1, 2=V4, 3=IT)
        self.region_labels = self._compute_region_labels()

    def _load_image_metadata(self, things_path: Path):
        """Load image class names and paths from things_imgs.mat."""
        split_key = "train_imgs" if self.split == "train" else "test_imgs"

        self.image_classes = []
        self.image_paths = []

        with h5py.File(things_path, "r") as f:
            group = f[split_key]
            class_refs = group["class"]
            path_refs = group["things_path"]
            n = class_refs.shape[0]

            for i in range(n):
                self.image_classes.append(_deref_cell_array(f, class_refs, i))
                self.image_paths.append(
                    _deref_cell_array(f, path_refs, i).replace("\\", "/")
                )

    def _compute_electrode_mask(self) -> np.ndarray:
        """Compute boolean mask for active electrodes based on region & SNR filters."""
        mask = np.ones(self.n_electrodes, dtype=bool)

        # Region filter
        if self.regions is not None:
            region_mask = np.zeros(self.n_electrodes, dtype=bool)
            region_ranges = REGION_MAP[self.monkey]
            for region in self.regions:
                if region not in region_ranges:
                    raise ValueError(
                        f"Unknown region {region} for {self.monkey}. "
                        f"Available: {list(region_ranges.keys())}"
                    )
                start, end = region_ranges[region]
                region_mask[start:end] = True
            mask &= region_mask

        # SNR filter
        if self.snr_threshold is not None:
            mask &= self.snr_max >= self.snr_threshold

        return mask

    def _compute_region_labels(self) -> np.ndarray:
        """Compute region label for each active electrode."""
        labels = np.zeros(self.n_electrodes, dtype=np.int64)
        region_ranges = REGION_MAP[self.monkey]
        for region_name, (start, end) in region_ranges.items():
            label = {"V1": 1, "V4": 2, "IT": 3}[region_name]
            labels[start:end] = label
        return labels[self.active_electrode_indices]

    def __len__(self) -> int:
        if self.split == "test_reps":
            return self.mua_data.shape[0] * self.mua_data.shape[1]
        return self.mua_data.shape[0]

    def _get_mua_vector(self, idx: int) -> np.ndarray:
        """Get the MUA vector for a given index, applying electrode mask."""
        if self.split == "test_reps":
            rep_idx = idx // self.mua_data.shape[1]
            img_idx = idx % self.mua_data.shape[1]
            mua = self.mua_data[rep_idx, img_idx, :]
        else:
            mua = self.mua_data[idx]

        return mua[self.active_electrode_indices]

    def _get_image_idx(self, idx: int) -> int:
        """Get the image index (for metadata lookup)."""
        if self.split == "test_reps":
            return idx % self.mua_data.shape[1]
        return idx

    def __getitem__(self, idx: int) -> Dict:
        mua = self._get_mua_vector(idx)
        img_idx = self._get_image_idx(idx)

        if self.mode == "raw":
            return {
                "mua": torch.from_numpy(mua),
                "image_class": self.image_classes[img_idx],
                "image_path": self.image_paths[img_idx],
                "region_labels": torch.from_numpy(self.region_labels),
                "electrode_indices": torch.from_numpy(
                    self.active_electrode_indices.astype(np.int64)
                ),
                "idx": idx,
            }

        elif self.mode == "capoyo":
            return self._make_capoyo_sample(mua, img_idx, idx)

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def _make_capoyo_sample(self, mua: np.ndarray, img_idx: int, idx: int) -> Dict:
        """Create a CaPOYO-compatible tokenized sample.

        For normMUA (time-averaged), we create 1 timepoint × N electrodes.
        Each electrode becomes an input token with:
          - value: MUA amplitude
          - unit_index: electrode global index
          - timestamp: 0.0 (single time point)
        """
        n = self.n_active_electrodes

        # Input tokens: one per electrode at time=0
        input_values = torch.from_numpy(mua).float().unsqueeze(-1)  # (N, 1)
        input_timestamps = torch.zeros(n, dtype=torch.float64)  # all at t=0
        input_unit_index = torch.arange(n, dtype=torch.int64)  # local indices

        # Latent tokens: fixed grid (single step for time-averaged data)
        # sequence_length will be set by the model, here we just provide metadata
        latent_step = 0.1  # 100ms (arbitrary for time-averaged data)
        num_latents_per_step = 8
        sequence_length = 0.1

        n_steps = max(1, int(sequence_length / latent_step))
        latent_timestamps = (
            np.arange(n_steps) * latent_step + latent_step / 2
        )
        latent_timestamps = np.repeat(latent_timestamps, num_latents_per_step)
        latent_index = np.tile(
            np.arange(num_latents_per_step, dtype=np.int64), n_steps
        )

        return {
            "input_values": input_values,  # (N, 1)
            "input_timestamps": input_timestamps,  # (N,)
            "input_unit_index": input_unit_index,  # (N,)
            "latent_index": torch.from_numpy(latent_index),
            "latent_timestamps": torch.from_numpy(latent_timestamps),
            "n_electrodes": n,
            "image_class": self.image_classes[img_idx],
            "image_path": self.image_paths[img_idx],
            "region_labels": torch.from_numpy(self.region_labels),
            "unit_ids": self.unit_ids.tolist(),
            "idx": idx,
        }

    def get_unit_ids(self) -> List[str]:
        """Return list of unit IDs for vocabulary initialization."""
        return self.unit_ids.tolist()

    def get_session_ids(self) -> List[str]:
        """Return list of session IDs."""
        return [f"{self.monkey}_normMUA"]

    def get_region_electrode_indices(self, region: str) -> np.ndarray:
        """Get indices of electrodes belonging to a specific brain region."""
        region_ranges = REGION_MAP[self.monkey]
        if region not in region_ranges:
            raise ValueError(f"Unknown region: {region}")
        start, end = region_ranges[region]
        region_indices = np.arange(start, end)
        # Map to active electrode indices
        return np.array(
            [
                np.where(self.active_electrode_indices == i)[0][0]
                for i in region_indices
                if i in self.active_electrode_indices
            ]
        )

    def get_data_stats(self) -> Dict:
        """Return statistics about the dataset for logging."""
        return {
            "monkey": self.monkey,
            "split": self.split,
            "n_samples": len(self),
            "n_electrodes": self.n_active_electrodes,
            "n_total_electrodes": self.n_electrodes,
            "regions": self.regions,
            "mua_mean": float(np.nanmean(self.mua_data[:, self.active_electrode_indices])),
            "mua_std": float(np.nanstd(self.mua_data[:, self.active_electrode_indices])),
            "mua_min": float(np.nanmin(self.mua_data[:, self.active_electrode_indices])),
            "mua_max": float(np.nanmax(self.mua_data[:, self.active_electrode_indices])),
            "n_classes": len(set(self.image_classes)),
        }
