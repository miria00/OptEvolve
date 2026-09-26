"""Kermany2017-OCT patch loader (LOCAL cache only, no network).

The dataset (zacharielegault/Kermany2017-OCT) is fully cached at
V2/data/kermany2017_oct: splits train (108309) / test (1000), features
`image` (PIL mode L, uint8, height ~496, VARIABLE width) and `label`
(CNV/DME/DRUSEN/NORMAL). These are real clinical retinal OCT B-scans;
the speckle in them is real measurement noise, not synthetic.

`load_patches(n, size, split, seed)` deterministically selects n scans
by seed and crops each to a (size, size) float64 patch in [0, 1].
"""
from __future__ import annotations

import os
import hashlib
import re

# Offline + CPU pinning BEFORE the heavy imports. The cache is complete;
# never touch the network. The GPU belongs to the vLLM server.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data",
    "kermany2017_oct",
)
DATASET_NAME = "zacharielegault/Kermany2017-OCT"
LABEL_NAMES = ("CNV", "DME", "DRUSEN", "NORMAL")

_DATASET_CACHE: dict = {}


def patient_group_hash(image_path: str | None) -> str | None:
    """Group scans using the dataset's class-patient-scan filename convention.

    Ignore class so the same patient token cannot leak across diagnostic
    labels. Do not expose the filename token in measurement manifests.
    """
    if not image_path:
        return None
    match = re.fullmatch(r"[A-Za-z]+-(\d+)-\d+\.(?:jpeg|jpg|png)",
                         os.path.basename(image_path), re.IGNORECASE)
    if not match:
        return None
    return hashlib.sha256(("Kermany2017-OCT:" + match[1]).encode()).hexdigest()


def _get_split(split: str):
    """Load (and memoize) one split from the local cache. No network."""
    if split not in _DATASET_CACHE:
        from datasets import load_dataset  # deferred: heavy import

        ds = load_dataset(DATASET_NAME, cache_dir=CACHE_DIR)
        for s in ds:
            _DATASET_CACHE[s] = ds[s]
    if split not in _DATASET_CACHE:
        raise KeyError(f"split {split!r} not in dataset (have {sorted(_DATASET_CACHE)})")
    return _DATASET_CACHE[split]


def _crop(arr: np.ndarray, size: int, rng: np.random.Generator, crop: str) -> np.ndarray:
    """Crop a (H, W) uint8 array to (size, size); edge-pad if too small."""
    H, W = arr.shape
    if H < size or W < size:  # defensive; OCT scans are >= 496 tall
        pad_h, pad_w = max(size - H, 0), max(size - W, 0)
        arr = np.pad(arr, ((0, pad_h), (0, pad_w)), mode="edge")
        H, W = arr.shape
    if crop == "center":
        top, left = (H - size) // 2, (W - size) // 2
    elif crop == "random":
        top = int(rng.integers(0, H - size + 1))
        left = int(rng.integers(0, W - size + 1))
    else:
        raise ValueError(f"crop must be 'center' or 'random', got {crop!r}")
    return arr[top : top + size, left : left + size]


def load_patches(
    n: int,
    size: int = 128,
    split: str = "test",
    seed: int = 0,
    crop: str = "center",
    return_meta: bool = False,
):
    """Load n real OCT patches from the LOCAL Kermany2017 cache.

    Selects n distinct scans deterministically by `seed`, converts each
    to float64 in [0, 1], and crops to (size, size) — center crop by
    default, seeded random crop with crop="random". Variable widths are
    handled by the crop (with defensive edge-padding for tiny scans).

    Returns patches of shape (n, size, size), dtype float64, values in
    [0, 1]. With return_meta=True also returns a list of per-patch dicts
    {index, label, label_name, orig_height, orig_width, source_split,
     patient_group_sha256}. Patient groups use the encoded image path;
    this function still samples images, not uniformly sampled patients.
    """
    if n <= 0:
        raise ValueError("n must be >= 1")
    if size <= 0:
        raise ValueError("size must be >= 1")
    ds = _get_split(split)
    if n > len(ds):
        raise ValueError(f"n={n} exceeds split size {len(ds)}")
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(ds), size=n, replace=False)

    patches = np.empty((n, size, size), dtype=np.float64)
    meta = []
    if return_meta:
        from datasets import Image
        encoded = ds.cast_column("image", Image(decode=False))
    for i, idx in enumerate(indices):
        row = ds[int(idx)]
        arr = np.asarray(row["image"], dtype=np.uint8)
        if arr.ndim != 2:  # mode L => 2D; defensive
            arr = arr[..., 0]
        patch = _crop(arr, size, rng, crop)
        patches[i] = patch.astype(np.float64) / 255.0
        if return_meta:
            label = int(row["label"])
            meta.append(
                {
                    "index": int(idx),
                    "label": label,
                    "label_name": LABEL_NAMES[label],
                    "orig_height": int(arr.shape[0]),
                    "orig_width": int(arr.shape[1]),
                    "source_split": split,
                    "patient_group_sha256": patient_group_hash(
                        encoded[int(idx)]["image"].get("path")),
                }
            )
    return (patches, meta) if return_meta else patches
