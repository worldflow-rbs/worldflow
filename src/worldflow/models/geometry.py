# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Per-point geometry positional encoding for the object-flow modality.

Each object-flow point receives an additive positional encoding derived from its
FRAME-0 absolute xyz (gravity-aligned, raw meters). The encoding is constant across
all F flow frames of a given point (frame-0 anchored) and is added to that point's
flow-token embedding in every frame.

Locked design::

    point_pe[i] = MLP_fuse( concat[ MLP_a(FoPE(xyz_i / abs_scale)),
                                    MLP_r(FoPE((xyz_i - centroid) / rel_scale)) ] )

where:
  * FoPE is a FIXED (non-learnable) NeRF-style sinusoidal Fourier encoding with
    ``num_freqs`` bands, mapping a 3-vector to ``3 * num_freqs * 2`` dims.
  * Branch a is world-absolute: divides by ``abs_scale`` (default 2.0 m, the world
    position range).
  * Branch r is centroid-relative: subtracts the per-sample centroid (mean over the
    N points) and divides by ``rel_scale`` (default 0.3 m, object-radius-normalized).
    Branch r also disambiguates points that share a coarse latent patch.
  * Fuse concatenates the two branch outputs (NOT sum: world scale ~2 m would drown
    object geometry ~0.3 m if summed) and projects to ``hidden_size``.

The 2.0 and 0.3 divisors are empirically-fixed constants exposed as module attributes.
"""

import torch
from torch import nn


def fourier_encode(coords: torch.Tensor, num_freqs: int) -> torch.Tensor:
    """Fixed NeRF-style sinusoidal Fourier positional encoding (FoPE).

    For each input coordinate and each frequency band ``l`` in ``0..num_freqs-1``,
    emits ``sin(2^l * pi * x)`` and ``cos(2^l * pi * x)``. The output concatenates over
    coordinates then bands then (sin, cos), giving ``coords.shape[-1] * num_freqs * 2``
    features. This is a pure function with NO learnable parameters and NO buffers.

    Args:
        coords: Input coordinates ``[..., C]`` (e.g. ``[N, 3]``), already scaled.
        num_freqs: Number of frequency bands ``L`` (NeRF frequencies ``2^l * pi``).

    Returns:
        Encoded tensor ``[..., C * num_freqs * 2]``.
    """
    # NeRF frequencies 2^l for l in 0..L-1, times pi.
    bands = (2.0 ** torch.arange(num_freqs, device=coords.device, dtype=coords.dtype)) * torch.pi  # [L]
    # coords: [..., C] -> [..., C, 1] * [L] -> [..., C, L]
    scaled = coords.unsqueeze(-1) * bands  # [...,C,L]
    # Interleave sin/cos on a trailing axis so layout is (coord, band, {sin,cos}).
    sin = torch.sin(scaled)  # [...,C,L]
    cos = torch.cos(scaled)  # [...,C,L]
    stacked = torch.stack([sin, cos], dim=-1)  # [...,C,L,2]
    return stacked.reshape(*coords.shape[:-1], coords.shape[-1] * num_freqs * 2)  # [...,C*L*2]


class ObjectFlowPointPE(nn.Module):
    """Additive per-point geometry PE for object-flow tokens.

    Consumes a single sample's frame-0 points ``[N, 3]`` (RAW xyz meters) and returns
    ``[N, hidden_size]``. The module is per-point and frame-independent; the caller
    broadcasts the result across all F frames.

    Args:
        hidden_size: Output dimension (matches the transformer hidden size).
        num_freqs: FoPE frequency bands (default 10).
        abs_scale: World-absolute divisor in meters (default 2.0, fixed).
        rel_scale: Centroid-relative divisor in meters (default 0.3, fixed).
    """

    def __init__(
        self,
        hidden_size: int,
        num_freqs: int = 10,
        abs_scale: float = 2.0,
        rel_scale: float = 0.3,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_freqs = num_freqs
        # Empirically-fixed scale constants (checkpoint-visible via network config too).
        self.abs_scale = abs_scale
        self.rel_scale = rel_scale

        fope_dim = 3 * num_freqs * 2  # 3 coords * L bands * (sin, cos)
        self.fope_dim = fope_dim
        # Each branch projects FoPE(3-vec) -> branch_dim. Fuse concat -> hidden.
        self.branch_dim = hidden_size // 2

        self.mlp_a = nn.Sequential(
            nn.Linear(fope_dim, self.branch_dim),
            nn.GELU(),
            nn.Linear(self.branch_dim, self.branch_dim),
        )
        self.mlp_r = nn.Sequential(
            nn.Linear(fope_dim, self.branch_dim),
            nn.GELU(),
            nn.Linear(self.branch_dim, self.branch_dim),
        )
        self.mlp_fuse = nn.Sequential(
            nn.Linear(2 * self.branch_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, points_xyz: torch.Tensor) -> torch.Tensor:
        """Compute the additive per-point PE for one sample.

        Args:
            points_xyz: Frame-0 absolute xyz for the N points ``[N, 3]`` (raw meters).

        Returns:
            Per-point positional encoding ``[N, hidden_size]``.
        """
        assert points_xyz.dim() == 2 and points_xyz.shape[-1] == 3, (
            f"ObjectFlowPointPE expects [N, 3], got {tuple(points_xyz.shape)}"
        )
        # Branch a: world-absolute.
        enc_a = fourier_encode(points_xyz / self.abs_scale, self.num_freqs)  # [N,fope_dim]
        # Branch r: centroid-relative.
        centroid = points_xyz.mean(dim=0, keepdim=True)  # [1,3]
        enc_r = fourier_encode((points_xyz - centroid) / self.rel_scale, self.num_freqs)  # [N,fope_dim]

        branch_a = self.mlp_a(enc_a)  # [N,branch_dim]
        branch_r = self.mlp_r(enc_r)  # [N,branch_dim]
        fused = torch.cat([branch_a, branch_r], dim=-1)  # [N,2*branch_dim]
        return self.mlp_fuse(fused)  # [N,hidden_size]
