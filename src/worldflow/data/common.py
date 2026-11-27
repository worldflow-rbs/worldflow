# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Temporal alignment, video preprocessing and query geometry shared by source adapters."""

import av
import numpy as np
import torch
from torch.nn import functional as F

from .points import select_object_indices

FRAMES, HEIGHT, WIDTH, POINTS = 49, 288, 480, 64


def evenly_spaced(length):
    if length < 1:
        raise ValueError("empty clip")
    return np.rint(np.linspace(0, length - 1, FRAMES)).astype(np.int64)


def decode_rgb(path, indices=None, native_resize=False):
    """Return aligned RGB frames and original dimensions. Count decoded frames for uniform sampling."""
    with av.open(path) as container:
        stream = container.streams.video[0]
        native_hw = stream.height, stream.width
        if indices is None:
            length = sum(1 for _ in container.decode(stream))
            indices = evenly_spaced(length)
            container.seek(0, stream=stream)
        requested = set(map(int, indices))
        if min(requested) < 0:
            raise ValueError("negative source frame index")
        wanted = {}
        for i, frame in enumerate(container.decode(stream)):
            if i in requested:
                if native_resize:
                    frame = frame.reformat(width=WIDTH, height=HEIGHT, format="rgb24", interpolation="BICUBIC")
                wanted[i] = frame.to_ndarray(format="rgb24")
                if len(wanted) == len(requested):
                    break
        if requested != wanted.keys():
            raise ValueError("video is shorter than its annotations")
    return np.stack([wanted[int(i)] for i in indices]), native_hw


def training_rgb(frames):
    if frames.ndim != 4 or frames.shape[0] != FRAMES or frames.shape[-1] != 3 or frames.dtype != np.uint8:
        raise ValueError("expected 49 aligned RGB uint8 frames")
    # Limit host memory per worker when downsampling native-resolution video.
    result = []
    for chunk in np.array_split(frames, 7):
        rgb = torch.from_numpy(np.ascontiguousarray(chunk)).permute(0, 3, 1, 2).float().div_(255)
        if rgb.shape[-2:] != (HEIGHT, WIDTH):
            rgb = F.interpolate(rgb, (HEIGHT, WIDTH), mode="bicubic", align_corners=False, antialias=True)
        normalized = ((rgb - 0.5) / 0.5).bfloat16().float()
        result.append(((normalized * 0.5 + 0.5) * 255).round().clamp_(0, 255).byte().permute(0, 2, 3, 1))
    return torch.cat(result).contiguous()


def canonical_geometry(tracks, uv, native_hw, identity, seed=42, source_points=None):
    """Select visible rows once for XYZ, the complete trajectory, and the observed pixel."""
    tracks = np.asarray(tracks, dtype=np.float32)
    uv = np.asarray(uv, dtype=np.float64)
    if tracks.ndim != 3 or tracks.shape[0] != FRAMES or tracks.shape[-1] != 3:
        raise ValueError("expected tracks (49, N, 3) in the first camera frame")
    first = np.asarray(tracks[0] if source_points is None else source_points, dtype=np.float32)
    if first.shape != tracks.shape[1:] or uv.shape != (len(first), 2):
        raise ValueError("source points, pixels and tracks must share their point axis")
    valid = np.isfinite(tracks).all(axis=(0, 2)) & np.isfinite(first).all(axis=1) & np.isfinite(uv).all(axis=1)
    tracks, first, uv = tracks[:, valid], first[valid], uv[valid]
    selection = select_object_indices(first, POINTS, seed=seed, sample_id=identity)
    if selection.indices is None:
        raise ValueError(f"{identity}: {selection.reason}")
    pick = selection.indices
    height, width = native_hw
    u = np.rint(uv[pick, 0] * WIDTH / width).clip(0, WIDTH - 1).astype(np.int64)
    v = np.rint(uv[pick, 1] * HEIGHT / height).clip(0, HEIGHT - 1).astype(np.int64)
    return {
        "points_xyz": torch.from_numpy(first[pick].copy()),
        "flow_m": torch.from_numpy(np.diff(tracks[:, pick], axis=0).copy()),
        "pixel_indices": torch.from_numpy(v * WIDTH + u),
    }
