# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Joint RGB video and metric 3D displacement inference."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .presets import DOMAIN_SCALES, SOURCE_DOMAINS, format_prompt

HEIGHT, WIDTH = 288, 480
FRAMES, POINTS = 49, 64


def _select_points(rgb, points_xyz, pixels_uv):
    if rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) < 1 or rgb.dtype != np.uint8:
        raise ValueError("rgb must be a non-empty uint8 array of shape (H, W, 3)")
    height, width = rgb.shape[:2]
    points = np.asarray(points_xyz, dtype=np.float32)
    uv = np.asarray(pixels_uv, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 1:
        raise ValueError("points_xyz must have shape (N, 3) with N >= 1")
    if uv.shape != (len(points), 2):
        raise ValueError("pixels_uv must have shape (N, 2) with one pixel per query point")
    if not np.isfinite(points).all() or not np.isfinite(uv).all():
        raise ValueError("query points and pixels must be finite")
    if (uv < 0).any() or (uv[:, 0] >= width).any() or (uv[:, 1] >= height).any():
        raise ValueError("pixels_uv contains a coordinate outside the input image")

    count = len(points)
    if count >= POINTS:
        selected = np.linspace(0, count - 1, POINTS).astype(np.int64)
    else:
        padding = np.linspace(0, count - 1, POINTS - count).astype(np.int64)
        selected = np.concatenate((np.arange(count, dtype=np.int64), padding))
    uv = uv[selected]
    u = np.rint(uv[:, 0] * (WIDTH / width)).clip(0, WIDTH - 1).astype(np.int64)
    v = np.rint(uv[:, 1] * (HEIGHT / height)).clip(0, HEIGHT - 1).astype(np.int64)
    return points[selected], v * WIDTH + u, selected


def _prepare_frame(rgb):
    # Match the reference bicubic resize and bf16 normalized pixel round trip.
    image = torch.from_numpy(rgb.copy(order="C")).float().permute(2, 0, 1)[None] / 255.0
    image = F.interpolate(image, size=(HEIGHT, WIDTH), mode="bicubic", align_corners=False, antialias=True)
    normalized = ((image[0] - 0.5) / 0.5).to(torch.bfloat16).float()
    return ((normalized * 0.5 + 0.5) * 255.0).round().clamp(0, 255).to(torch.uint8)


def _build_batch(model, rgb, points, flat_pixels, prompt):
    return {
        "frame": _prepare_frame(rgb).to(model.device),
        "points_xyz": torch.as_tensor(points, device=model.device),
        "pixel_indices": torch.as_tensor(flat_pixels, device=model.device),
        "prompt": prompt,
        "num_frames": FRAMES,
    }


def predict(
    checkpoint: Path,
    rgb: np.ndarray,
    points_xyz: np.ndarray,
    pixels_uv: np.ndarray,
    instruction: str,
    source: str,
    steps: int = 40,
    guidance: float = 2.0,
    shift: float = 5.0,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Predict 49 RGB frames and 48 displacements from the first-frame points.

    Input XYZ is in meters in the initial camera frame. Corresponding (u, v)
    pixels use the original image resolution. The returned selected_indices
    identify the 64 input rows used for predictions, including any duplicates.
    """
    if not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer")
    if not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    if not math.isfinite(guidance) or not math.isfinite(shift) or shift <= 0:
        raise ValueError("guidance must be finite and shift must be finite and positive")
    prompt = format_prompt(instruction, source)
    xyz, flat_pixels, selected = _select_points(rgb, points_xyz, pixels_uv)

    from .loader import load_model

    model = load_model(Path(checkpoint))
    with torch.inference_mode():
        batch = _build_batch(model, rgb, xyz, flat_pixels, prompt)
        generated = model.generate_samples_from_batch(
            batch, seed=[seed], num_steps=steps, guidance=guidance, shift=shift
        )
        flow = generated["objectflow"][0].detach().float().cpu().numpy()
        if flow.shape != (FRAMES - 1, POINTS, 3) or not np.isfinite(flow).all():
            raise RuntimeError("expected finite generated flow with shape (48, 64, 3)")
        scale = DOMAIN_SCALES[SOURCE_DOMAINS[source]]
        offsets = np.cumsum(flow * np.float32(scale), axis=0)
        if not np.isfinite(offsets).all():
            raise RuntimeError("non-finite metric displacements")
        decoded = model.decode(generated["vision"][0]).detach().float().cpu()
        if decoded.ndim == 5 and decoded.shape[0] == 1:
            decoded = decoded[0]
        if decoded.shape != (3, FRAMES, HEIGHT, WIDTH) or not torch.isfinite(decoded).all():
            raise RuntimeError("expected finite decoded video with shape (3, 49, 288, 480)")
        frames = decoded.permute(1, 2, 3, 0)
        video = ((frames * 0.5 + 0.5) * 255.0).round().clamp_(0, 255).to(torch.uint8)
    return {
        "video": video.contiguous().numpy(),
        "displacements": offsets,
        "points_xyz": xyz,
        "selected_indices": selected,
    }
