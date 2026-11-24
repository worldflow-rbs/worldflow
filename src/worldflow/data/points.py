# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Deterministic object-points-v2 selection for visible metric query points."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

# Versioned so stats/audit snapshots can prove they used the same point
# selection contract as the training adapters.
OBJECT_POINT_POLICY_VERSION = "object-points-v2"


@dataclass(frozen=True)
class ObjectPointSelection:
    """Result of the shared ``object-points-v2`` selection policy.

    ``indices`` indexes the original candidate point cloud.  It is ``None`` when
    the candidate cloud is rejected (currently only the ``n < min_points`` case).
    ``replacement_count`` counts duplicate draws in the returned index array and
    ``unique_count`` is the number of distinct source points represented there.
    """

    indices: np.ndarray | None
    replacement_count: int
    selection_mode: str
    unique_count: int
    reason: str | None = None


def _stable_seed(seed: int, sample_id: str) -> int:
    """Derive a process-independent uint64 seed from a seed and sample identity."""
    payload = f"{int(seed)}\0{sample_id}".encode("utf-8", errors="surrogatepass")
    digest = hashlib.blake2b(payload, digest_size=8, person=b"obj-points-v2").digest()
    return int.from_bytes(digest, byteorder="little", signed=False)


def _fnv_hash_vec(arr: np.ndarray) -> np.ndarray:
    """FNV-1a hash over integer voxel-coordinate vectors."""
    out = np.full(arr.shape[0], np.uint64(14695981039346656037), dtype=np.uint64)
    for j in range(arr.shape[1]):
        out = out * np.uint64(1099511628211)
        out = out ^ arr[:, j].astype(np.uint64)
    return out


def _voxel_select(points: np.ndarray, grid_size: float, mode: str = "test") -> np.ndarray:
    """Pick one point per voxel. mode="test": nearest-to-centroid (deterministic);
    mode="train": random offset within voxel. Returns the chosen point indices."""
    scaled = points / grid_size
    grid_coord = np.floor(scaled).astype(np.int64)
    grid_coord -= grid_coord.min(axis=0)
    keys = _fnv_hash_vec(grid_coord)
    idx_sort = np.argsort(keys)
    keys_sorted = keys[idx_sort]
    _, _inv, counts = np.unique(keys_sorted, return_inverse=True, return_counts=True)
    cumsum = np.cumsum(np.insert(counts, 0, 0)[:-1])
    if mode == "train":
        offsets = np.random.randint(0, counts.max(), size=counts.shape) % counts
        return idx_sort[cumsum + offsets]
    # nearest-to-centroid (deterministic)
    pts_sorted = points[idx_sort].astype(np.float64)
    sums = np.add.reduceat(pts_sorted, cumsum, axis=0)
    centroids = sums / counts[:, None]
    centroid_per_point = np.repeat(centroids, counts, axis=0)
    d2 = np.sum((pts_sorted - centroid_per_point) ** 2, axis=1)
    d2_min_per_voxel = np.minimum.reduceat(d2, cumsum)
    d2_min_per_point = np.repeat(d2_min_per_voxel, counts)
    is_min = d2 == d2_min_per_point
    zero_pos = np.where(is_min)[0]
    group_of_zero = np.searchsorted(cumsum, zero_pos, side="right") - 1
    _, first_pos = np.unique(group_of_zero, return_index=True)
    selected_local = zero_pos[first_pos]
    return idx_sort[selected_local]


def _voxel_count(points: np.ndarray, grid_size: float) -> int:
    """Occupied-voxel COUNT at `grid_size` — same first lines as _voxel_select, then just a
    hash+unique. Equals len(_voxel_select(...)) by construction (one point per voxel), so
    bisecting on it walks the exact same grid-size trajectory without the argsort/centroid math."""
    scaled = points / grid_size
    grid_coord = np.floor(scaled).astype(np.int64)
    grid_coord -= grid_coord.min(axis=0)
    return len(np.unique(_fnv_hash_vec(grid_coord)))


def voxel_uniform_idx(points: np.ndarray, num_points: int, mode: str = "test") -> np.ndarray:
    """Voxel grid downsample (one point per voxel, nearest-to-centroid) then linspace to exactly
    num_points. Bisects the grid size to land near 1.5*num_points occupied voxels. The bisection
    needs only the occupied-voxel COUNT, so it runs the hash-only _voxel_count (identical
    trajectory: same lo/hi rule, 30-iteration cap, 5% early break as the canonical sampler) and
    remembers the grid size whose count lands CLOSEST to target_vox (strict < keeps the first on
    ties -> deterministic); the full _voxel_select runs exactly ONCE, at that best-seen grid size.

    Returns exactly num_points indices into `points`. When occupied voxels < num_points, the voxel
    reps are deterministically topped up (linspace with repeats) to reach the target — so it never
    fails on small clouds. Empty input -> zeros."""
    points = np.asarray(points, np.float64)
    num_points = int(num_points)
    if points.shape[0] == 0 or num_points <= 0:
        return np.zeros(max(num_points, 0), dtype=np.int64)
    target_vox = int(num_points * 1.5)
    lo, hi = 0.001, 10.0
    best_grid, best_err = None, None
    for _ in range(30):
        mid = (lo + hi) / 2
        n_vox = _voxel_count(points, mid)
        err = abs(n_vox - target_vox)
        if best_err is None or err < best_err:  # first-seen wins ties
            best_grid, best_err = mid, err
        if n_vox > target_vox:
            lo = mid
        else:
            hi = mid
        if err / max(target_vox, 1) < 0.05:
            break
    best_idx = (
        _voxel_select(points, best_grid, mode) if best_grid is not None else np.zeros(0, dtype=np.int64)
    )  # single full select
    n_vox = len(best_idx)
    if n_vox == 0:
        return np.zeros(num_points, dtype=np.int64)
    if n_vox >= num_points:
        sub = np.linspace(0, n_vox - 1, num_points).astype(np.int64)  # uniform trim of voxel reps
        return best_idx[sub]
    extra = np.linspace(0, n_vox - 1, num_points - n_vox).astype(np.int64)  # deterministic top-up
    return np.concatenate([best_idx, best_idx[extra]])


def select_object_indices(
    points: np.ndarray,
    num_points: int = 64,
    *,
    min_points: int = 8,
    seed: int = 0,
    sample_id: str = "",
) -> ObjectPointSelection:
    """Select object-point indices according to the ``object-points-v2`` policy.

    The policy is intentionally independent of a dataset adapter's preceding
    validity/outlier filtering; ``points`` must already be the candidate cloud
    for that sample.  Selection is:

    * ``n < min_points``: reject the sample;
    * ``min_points <= n < num_points``: stable random sampling *with*
      replacement, seeded by ``seed`` and ``sample_id``;
    * ``n == num_points``: identity;
    * ``n > num_points``: deterministic voxel + uniform sampling.

    ``indices`` always has exactly ``num_points`` entries for an accepted sample.
    ``replacement_count`` is zero for identity and downsampling, and is the
    number of duplicate draws for the replacement path.  ``reason`` is
    ``"too_few_points"`` for a rejected sample and ``None`` otherwise.
    """
    points = np.asarray(points)
    if points.ndim == 0:
        raise ValueError("points must be an array with a point dimension")
    n = int(points.shape[0])
    k = int(num_points)
    minimum = int(min_points)
    if k <= 0:
        raise ValueError(f"num_points must be positive, got {num_points!r}")
    if minimum <= 0 or minimum > k:
        raise ValueError(f"min_points must satisfy 1 <= min_points <= num_points, got {min_points!r}")

    if n < minimum:
        return ObjectPointSelection(
            indices=None,
            replacement_count=0,
            selection_mode="reject",
            unique_count=0,
            reason="too_few_points",
        )

    if n == k:
        idx = np.arange(n, dtype=np.int64)
        return ObjectPointSelection(idx, 0, "identity", n, None)

    if n < k:
        rng = np.random.default_rng(_stable_seed(seed, str(sample_id)))
        idx = rng.choice(n, size=k, replace=True).astype(np.int64, copy=False)
        unique_count = int(np.unique(idx).size)
        return ObjectPointSelection(
            indices=idx,
            replacement_count=k - unique_count,
            selection_mode="random_replacement",
            unique_count=unique_count,
            reason=None,
        )

    idx = np.asarray(voxel_uniform_idx(points, k, mode="test"), dtype=np.int64)
    unique_count = int(np.unique(idx).size)
    return ObjectPointSelection(
        indices=idx,
        replacement_count=k - unique_count,
        selection_mode="voxel_uniform",
        unique_count=unique_count,
        reason=None,
    )
