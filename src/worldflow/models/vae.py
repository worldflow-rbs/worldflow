# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1 AND Apache-2.0
"""Wan 2.2 causal video autoencoder used for WorldFlow's RGB stream.

Adapted from Cosmos Framework for local, single-device inference.
"""

from pathlib import Path

import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F

# For sequential decoding, CACHE_T is the number of frames to cache.
CACHE_T = 2


def _contiguous_clone(t: torch.Tensor) -> torch.Tensor:
    """Return a contiguous copy of *t* using exactly one allocation.

    When *t* is already contiguous, ``.contiguous()`` would be a no-op that
    returns the *same* tensor (sharing storage), so we need ``.clone()``.
    When *t* is non-contiguous, ``.contiguous()`` already allocates a fresh
    tensor with independent storage — no extra ``.clone()`` needed.
    """
    if t.is_contiguous():
        return t.clone()
    return t.contiguous()


def _update_cache_and_apply(
    x: torch.Tensor,
    layer: "CausalConv3d",
    feat_cache: list,
    feat_idx: list[int],
) -> torch.Tensor:
    """Apply a CausalConv3d with temporal cache management.

    Saves the last CACHE_T frames of ``x`` as the new cache entry and,
    when the current chunk has fewer than 2 frames, prepends the last
    cached frame so the cache always spans 2 frames.

    Note that feat_idx is a list with a single element, which stores
    the index of the current CausalConv3d layer. List is used here so
    feat_idx can be mutated in place, and the caller can pass in a reference
    to the list.
    """
    idx = feat_idx[0]
    cache_x = _contiguous_clone(x[:, :, -CACHE_T:, :, :])
    if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
        cache_x = torch.cat(
            [
                feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device),
                cache_x,
            ],
            dim=2,
        )  # [B,C,2,H,W]
    x = layer(x, feat_cache[idx])
    feat_cache[idx] = cache_x
    feat_idx[0] += 1
    return x


class CausalConv3d(nn.Conv3d):
    """
    Causal 3d convolution.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._padding = (
            self.padding[2],
            self.padding[2],
            self.padding[1],
            self.padding[1],
            2 * self.padding[0],
            0,
        )
        self.padding = (0, 0, 0)

    def forward(self, x, cache_x=None):  # x: [B,C,T,H,W]
        padding = list(self._padding)
        if cache_x is not None and self._padding[4] > 0:
            cache_x = cache_x.to(x.device)
            x = torch.cat([cache_x, x], dim=2)  # [B,C,T+cache_T,H,W]
            padding[4] -= cache_x.shape[2]
        x = F.pad(x, padding)  # [B,C,T_padded,H_padded,W_padded]

        return super().forward(x)  # [B,out_C,T_out,H_out,W_out]


class RMS_norm(nn.Module):
    def __init__(self, dim, channel_first=True, images=True, bias=False):
        super().__init__()
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)

        self.channel_first = channel_first
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else 0.0

    def forward(self, x):
        return F.normalize(x, dim=(1 if self.channel_first else -1)) * self.scale * self.gamma + self.bias


class Upsample(nn.Upsample):
    def forward(self, x):
        """
        Fix bfloat16 support for nearest neighbor interpolation.
        """
        return super().forward(x.float()).type_as(x)


class Resample(nn.Module):
    def __init__(self, dim, mode):
        assert mode in (
            "none",
            "upsample2d",
            "upsample3d",
            "downsample2d",
            "downsample3d",
        )
        super().__init__()
        self.dim = dim
        self.mode = mode

        # layers
        if mode == "upsample2d":
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                nn.Conv2d(dim, dim, 3, padding=1),
            )
        elif mode == "upsample3d":
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                nn.Conv2d(dim, dim, 3, padding=1),
            )
            self.time_conv = CausalConv3d(dim, dim * 2, (3, 1, 1), padding=(1, 0, 0))
        elif mode == "downsample2d":
            self.resample = nn.Sequential(nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(dim, dim, 3, stride=(2, 2)))
        elif mode == "downsample3d":
            self.resample = nn.Sequential(nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(dim, dim, 3, stride=(2, 2)))
            self.time_conv = CausalConv3d(dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))
        else:
            self.resample = nn.Identity()

    def forward(self, x, feat_cache=None, feat_idx=[0]):  # x: [B,C,T,H,W]
        b, c, t, h, w = x.size()
        if self.mode == "upsample3d":
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    # First frame: skip time_conv, seed cache with zeros so the next call sees a real tensor
                    feat_cache[idx] = torch.zeros(b, c, CACHE_T, h, w, device=x.device, dtype=x.dtype)  # [B,C,2,H,W]
                    feat_idx[0] += 1
                else:
                    cache_x = _contiguous_clone(x[:, :, -CACHE_T:, :, :])  # [B,C,<=2,H,W]
                    if cache_x.shape[2] < 2:
                        cache_x = torch.cat(
                            [
                                feat_cache[idx][:, :, -1, :, :].unsqueeze(2),
                                cache_x,
                            ],
                            dim=2,
                        )  # [B,C,2,H,W]
                    x = self.time_conv(x, feat_cache[idx])  # [B,C*2,T,H,W]
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1
                    x = x.reshape(b, 2, c, t, h, w)  # [B,2,C,T,H,W]
                    x = torch.stack((x[:, 0, :, :, :, :], x[:, 1, :, :, :, :]), 3)  # [B,C,T,2,H,W]
                    x = x.reshape(b, c, t * 2, h, w)  # [B,C,T*2,H,W]
        t = x.shape[2]
        x = rearrange(x, "b c t h w -> (b t) c h w")  # [B*T,C,H,W]
        x = self.resample(x)  # [B*T,C_out,H_out,W_out]
        x = rearrange(x, "(b t) c h w -> b c t h w", t=t)  # [B,C_out,T,H_out,W_out]

        if self.mode == "downsample3d":
            # Important for torch.compile: when we're *not* doing streaming/cache-based inference
            # (feat_cache is None), we still need to apply the temporal downsample conv.
            if feat_cache is None:
                # `time_conv` has kernel (3,1,1), stride 2 in time, and no internal temporal padding.
                # In the streaming path, we effectively provide left temporal context via cached frames.
                # For the non-streaming path, pad 2 frames on the left so:
                # - the conv is always valid (T>=3)
                # - the output temporal length matches the shortcut path's ceil(T/2) behavior
                x = F.pad(x, (0, 0, 0, 0, 2, 0))  # [B,C,T+2,H_out,W_out]
                x = self.time_conv(x)  # [B,C,T//2+1,H_out,W_out]
            else:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    # First call for this layer in a streaming/windowed pass.
                    # The baseline streaming path primes caches with a single-frame chunk (T==1),
                    # where skipping time_conv preserves both correctness and shape alignment.
                    #
                    # If this is ever called with T>1 (non-standard chunking), fall back to a padded
                    # time_conv so the main path stays compatible with the shortcut path.
                    if x.shape[2] == 1:
                        feat_cache[idx] = _contiguous_clone(x)
                    else:
                        cache_x = _contiguous_clone(x[:, :, -1:, :, :])  # [B,C,1,H_out,W_out]
                        x_in = F.pad(x, (0, 0, 0, 0, 2, 0))  # [B,C,T+2,H_out,W_out]
                        x = self.time_conv(x_in)  # [B,C,T//2+1,H_out,W_out]
                        feat_cache[idx] = cache_x
                    feat_idx[0] += 1
                else:
                    cache_x = _contiguous_clone(x[:, :, -1:, :, :])  # [B,C,1,H_out,W_out]
                    x_cat = torch.cat([feat_cache[idx][:, :, -1:, :, :], x], 2)  # [B,C,T+1,H_out,W_out]
                    t_cat = x_cat.shape[2]
                    if t_cat < 3:
                        x_cat = F.pad(x_cat, (0, 0, 0, 0, 3 - t_cat, 0))  # [B,C,3,H_out,W_out]
                    x = self.time_conv(x_cat)  # [B,C,T//2+1,H_out,W_out]
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        # layers
        self.residual = nn.Sequential(
            RMS_norm(in_dim, images=False),
            nn.SiLU(),
            CausalConv3d(in_dim, out_dim, 3, padding=1),
            RMS_norm(out_dim, images=False),
            nn.SiLU(),
            nn.Dropout(dropout),
            CausalConv3d(out_dim, out_dim, 3, padding=1),
        )
        self.shortcut = CausalConv3d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        h = self.shortcut(x)
        for layer in self.residual:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                x = _update_cache_and_apply(x, layer, feat_cache, feat_idx)
            else:
                x = layer(x)
        return x + h


class AttentionBlock(nn.Module):
    """
    Causal self-attention with a single head.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        # layers
        self.norm = RMS_norm(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

        # zero out the last layer params
        nn.init.zeros_(self.proj.weight)

    def forward(self, x):  # x: [B,C,T,H,W]
        identity = x
        b, c, t, h, w = x.size()
        x = rearrange(x, "b c t h w -> (b t) c h w")  # [B*T,C,H,W]
        x = self.norm(x)  # [B*T,C,H,W]
        # compute query, key, value
        q, k, v = self.to_qkv(x).reshape(b * t, 1, c * 3, -1).permute(0, 1, 3, 2).contiguous().chunk(3, dim=-1)
        # q,k,v: [B*T,1,H*W,C]

        # apply attention
        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
        )  # [B*T,1,H*W,C]
        x = x.squeeze(1).permute(0, 2, 1).contiguous().reshape(b * t, c, h, w)  # [B*T,C,H,W]

        # output
        x = self.proj(x)  # [B*T,C,H,W]
        x = rearrange(x, "(b t) c h w-> b c t h w", t=t)  # [B,C,T,H,W]
        return x + identity  # [B,C,T,H,W]


def patchify(x, patch_size):  # x: [B,C,H,W] or [B,C,T,H,W] -> [B,C*p^2,H//p,W//p] or [B,C*p^2,T,H//p,W//p]
    if patch_size == 1:
        return x
    # Fast path: patch_size==2 is the only one used in this tokenizer.
    # Implement it with pure view/permute/reshape to be maximally torch.compile-friendly.
    if patch_size == 2:
        if x.dim() == 4:
            b, c, h, w = x.shape
            x = x.view(b, c, h // 2, 2, w // 2, 2)  # [B,C,H//2,2,W//2,2]
            x = x.permute(0, 1, 5, 3, 2, 4).contiguous()  # [B,C,2,2,H//2,W//2]
            return x.view(b, c * 4, h // 2, w // 2)  # [B,C*4,H//2,W//2]
        if x.dim() == 5:
            b, c, f, h, w = x.shape
            x = x.view(b, c, f, h // 2, 2, w // 2, 2)  # [B,C,T,H//2,2,W//2,2]
            x = x.permute(0, 1, 6, 4, 2, 3, 5).contiguous()  # [B,C,2,2,T,H//2,W//2]
            return x.view(b, c * 4, f, h // 2, w // 2)  # [B,C*4,T,H//2,W//2]
    if x.dim() == 4:
        x = rearrange(x, "b c (h q) (w r) -> b (c r q) h w", q=patch_size, r=patch_size)  # [B,C*p^2,H//p,W//p]
    elif x.dim() == 5:
        x = rearrange(
            x,
            "b c f (h q) (w r) -> b (c r q) f h w",
            q=patch_size,
            r=patch_size,
        )  # [B,C*p^2,T,H//p,W//p]
    else:
        raise ValueError(f"Invalid input shape: {x.shape}")

    return x


def unpatchify(x, patch_size):  # x: [B,C*p^2,H,W] or [B,C*p^2,T,H,W] -> [B,C,H*p,W*p] or [B,C,T,H*p,W*p]
    if patch_size == 1:
        return x

    if x.dim() == 4:
        x = rearrange(x, "b (c r q) h w -> b c (h q) (w r)", q=patch_size, r=patch_size)  # [B,C,H*p,W*p]
    elif x.dim() == 5:
        x = rearrange(
            x,
            "b (c r q) f h w -> b c f (h q) (w r)",
            q=patch_size,
            r=patch_size,
        )  # [B,C,T,H*p,W*p]
    return x


class AvgDown3D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        factor_t,
        factor_s=1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor_t = factor_t
        self.factor_s = factor_s
        self.factor = self.factor_t * self.factor_s * self.factor_s

        assert in_channels * self.factor % out_channels == 0
        self.group_size = in_channels * self.factor // out_channels

    def forward(
        self, x: torch.Tensor
    ) -> torch.Tensor:  # x: [B,C,T,H,W] -> [B,out_channels,T//factor_t,H//factor_s,W//factor_s]
        pad_t = (self.factor_t - x.shape[2] % self.factor_t) % self.factor_t
        pad = (0, 0, 0, 0, pad_t, 0)
        x = F.pad(x, pad)  # [B,C,T_padded,H,W]
        B, C, T, H, W = x.shape
        x = x.view(
            B,
            C,
            T // self.factor_t,
            self.factor_t,
            H // self.factor_s,
            self.factor_s,
            W // self.factor_s,
            self.factor_s,
        )  # [B,C,T//ft,ft,H//fs,fs,W//fs,fs]
        x = x.permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous()  # [B,C,ft,fs,fs,T//ft,H//fs,W//fs]
        x = x.view(
            B,
            C * self.factor,
            T // self.factor_t,
            H // self.factor_s,
            W // self.factor_s,
        )  # [B,C*factor,T//ft,H//fs,W//fs]
        x = x.view(
            B,
            self.out_channels,
            self.group_size,
            T // self.factor_t,
            H // self.factor_s,
            W // self.factor_s,
        )  # [B,out_channels,group_size,T//ft,H//fs,W//fs]
        x = x.mean(dim=2)  # [B,out_channels,T//ft,H//fs,W//fs]
        return x


class DupUp3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        factor_t,
        factor_s=1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.factor_t = factor_t
        self.factor_s = factor_s
        self.factor = self.factor_t * self.factor_s * self.factor_s

        assert out_channels * self.factor % in_channels == 0
        self.repeats = out_channels * self.factor // in_channels

    def forward(
        self, x: torch.Tensor, first_chunk=False
    ) -> torch.Tensor:  # x: [B,in_channels,T,H,W] -> [B,out_channels,T*factor_t,H*factor_s,W*factor_s]
        x = x.repeat_interleave(self.repeats, dim=1)  # [B,in_channels*repeats,T,H,W]
        x = x.view(
            x.size(0),
            self.out_channels,
            self.factor_t,
            self.factor_s,
            self.factor_s,
            x.size(2),
            x.size(3),
            x.size(4),
        )  # [B,out_channels,ft,fs,fs,T,H,W]
        x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()  # [B,out_channels,T,ft,H,fs,W,fs]
        x = x.view(
            x.size(0),
            self.out_channels,
            x.size(2) * self.factor_t,
            x.size(4) * self.factor_s,
            x.size(6) * self.factor_s,
        )  # [B,out_channels,T*ft,H*fs,W*fs]
        if first_chunk:
            x = x[:, :, self.factor_t - 1 :, :, :]  # [B,out_channels,T*ft-ft+1,H*fs,W*fs]
        return x


class Down_ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout, mult, temperal_downsample=False, down_flag=False):
        super().__init__()

        # Shortcut path with downsample
        self.avg_shortcut = AvgDown3D(
            in_dim,
            out_dim,
            factor_t=2 if temperal_downsample else 1,
            factor_s=2 if down_flag else 1,
        )

        # Main path with residual blocks and downsample
        downsamples = []
        for _ in range(mult):
            downsamples.append(ResidualBlock(in_dim, out_dim, dropout))
            in_dim = out_dim

        # Add the final downsample block
        if down_flag:
            mode = "downsample3d" if temperal_downsample else "downsample2d"
            downsamples.append(Resample(out_dim, mode=mode))

        self.downsamples = nn.Sequential(*downsamples)

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        # Avoid cloning the full activation (extra kernel + bandwidth).
        # None of the downstream modules are in-place, so taking the shortcut first is safe.
        x_shortcut = self.avg_shortcut(x)
        for module in self.downsamples:
            x = module(x, feat_cache, feat_idx)

        return x + x_shortcut


class Up_ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout, mult, temperal_upsample=False, up_flag=False):
        super().__init__()
        # Shortcut path with upsample
        if up_flag:
            self.avg_shortcut = DupUp3D(
                in_dim,
                out_dim,
                factor_t=2 if temperal_upsample else 1,
                factor_s=2 if up_flag else 1,
            )
        else:
            self.avg_shortcut = None

        # Main path with residual blocks and upsample
        upsamples = []
        for _ in range(mult):
            upsamples.append(ResidualBlock(in_dim, out_dim, dropout))
            in_dim = out_dim

        # Add the final upsample block
        if up_flag:
            mode = "upsample3d" if temperal_upsample else "upsample2d"
            upsamples.append(Resample(out_dim, mode=mode))

        self.upsamples = nn.Sequential(*upsamples)

    def forward(self, x, feat_cache=None, feat_idx=[0], first_chunk=False):
        x_shortcut = self.avg_shortcut(x, first_chunk) if self.avg_shortcut is not None else None
        for module in self.upsamples:
            x = module(x, feat_cache, feat_idx)
        if x_shortcut is not None:
            return x + x_shortcut
        return x


class Encoder3d(nn.Module):
    def __init__(
        self,
        dim=128,
        z_dim=4,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_downsample=[False, True, True],
        dropout=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample

        # dimensions
        dims = [dim * u for u in [1] + dim_mult]
        scale = 1.0

        # init block
        self.conv1 = CausalConv3d(12, dims[0], 3, padding=1)

        # downsample blocks
        downsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            t_down_flag = temperal_downsample[i] if i < len(temperal_downsample) else False
            downsamples.append(
                Down_ResidualBlock(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    dropout=dropout,
                    mult=num_res_blocks,
                    temperal_downsample=t_down_flag,
                    down_flag=i != len(dim_mult) - 1,
                )
            )
            scale /= 2.0
        self.downsamples = nn.Sequential(*downsamples)

        # middle blocks
        self.middle = nn.Sequential(
            ResidualBlock(out_dim, out_dim, dropout),
            AttentionBlock(out_dim),
            ResidualBlock(out_dim, out_dim, dropout),
        )

        # output blocks
        self.head = nn.Sequential(
            RMS_norm(out_dim, images=False),
            nn.SiLU(),
            CausalConv3d(out_dim, z_dim, 3, padding=1),
        )

    def forward(self, x, feat_cache=None):  # x: [B,12,T,H//2,W//2] -> [B,z_dim,T//4,H//16,W//16]
        feat_idx = [0]

        if feat_cache is not None:
            x = _update_cache_and_apply(x, self.conv1, feat_cache, feat_idx)  # [B,dim,T,H//2,W//2]
        else:
            x = self.conv1(x)  # [B,dim,T,H//2,W//2]

        # downsamples
        for layer in self.downsamples:
            if feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)
        # x: [B,dim*dim_mult[-1],T//4,H//16,W//16]

        # middle
        for layer in self.middle:
            if isinstance(layer, ResidualBlock) and feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)
        # x: [B,dim*dim_mult[-1],T//4,H//16,W//16]

        # head
        for layer in self.head:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                x = _update_cache_and_apply(x, layer, feat_cache, feat_idx)
            else:
                x = layer(x)

        return x  # [B,z_dim,T//4,H//16,W//16]


class Decoder3d(nn.Module):
    def __init__(
        self,
        dim=128,
        z_dim=4,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_upsample=[True, True, False],
        dropout=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_upsample = temperal_upsample

        # dimensions
        dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]

        # init block
        self.conv1 = CausalConv3d(z_dim, dims[0], 3, padding=1)

        # middle blocks
        self.middle = nn.Sequential(
            ResidualBlock(dims[0], dims[0], dropout),
            AttentionBlock(dims[0]),
            ResidualBlock(dims[0], dims[0], dropout),
        )

        # upsample blocks
        upsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            t_up_flag = temperal_upsample[i] if i < len(temperal_upsample) else False
            upsamples.append(
                Up_ResidualBlock(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    dropout=dropout,
                    mult=num_res_blocks + 1,
                    temperal_upsample=t_up_flag,
                    up_flag=i != len(dim_mult) - 1,
                )
            )
        self.upsamples = nn.Sequential(*upsamples)

        # output blocks
        self.head = nn.Sequential(
            RMS_norm(out_dim, images=False),
            nn.SiLU(),
            CausalConv3d(out_dim, 12, 3, padding=1),
        )

    def forward(self, x, feat_cache=None, first_chunk=False):  # x: [B,z_dim,T,H,W] -> [B,12,T*4,H*8,W*8]
        feat_idx = [0]

        if feat_cache is not None:
            x = _update_cache_and_apply(x, self.conv1, feat_cache, feat_idx)  # [B,dim*dim_mult[-1],T,H,W]
        else:
            x = self.conv1(x)  # [B,dim*dim_mult[-1],T,H,W]

        for layer in self.middle:
            if isinstance(layer, ResidualBlock) and feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)
        # x: [B,dim*dim_mult[-1],T,H,W]

        # upsamples
        for layer in self.upsamples:
            if feat_cache is not None:
                x = layer(x, feat_cache, feat_idx, first_chunk)
            else:
                x = layer(x)
        # x: [B,dec_dim,T*4,H*8,W*8]

        # head
        for layer in self.head:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                x = _update_cache_and_apply(x, layer, feat_cache, feat_idx)
            else:
                x = layer(x)
        return x  # [B,12,T*4,H*8,W*8]


def count_conv3d(model: nn.Module) -> int:
    return sum(1 for m in model.modules() if isinstance(m, CausalConv3d))


_LATENT_MEAN = (
    -0.2289,
    -0.0052,
    -0.1323,
    -0.2339,
    -0.2799,
    0.0174,
    0.1838,
    0.1557,
    -0.1382,
    0.0542,
    0.2813,
    0.0891,
    0.157,
    -0.0098,
    0.0375,
    -0.1825,
    -0.2246,
    -0.1207,
    -0.0698,
    0.5109,
    0.2665,
    -0.2108,
    -0.2158,
    0.2502,
    -0.2055,
    -0.0322,
    0.1109,
    0.1567,
    -0.0729,
    0.0899,
    -0.2799,
    -0.123,
    -0.0313,
    -0.1649,
    0.0117,
    0.0723,
    -0.2839,
    -0.2083,
    -0.052,
    0.3748,
    0.0152,
    0.1957,
    0.1433,
    -0.2944,
    0.3573,
    -0.0548,
    -0.1681,
    -0.0667,
)
_LATENT_STD = (
    0.4765,
    1.0364,
    0.4514,
    1.1677,
    0.5313,
    0.499,
    0.4818,
    0.5013,
    0.8158,
    1.0344,
    0.5894,
    1.0901,
    0.6885,
    0.6165,
    0.8454,
    0.4978,
    0.5759,
    0.3523,
    0.7135,
    0.6804,
    0.5833,
    1.4146,
    0.8986,
    0.5659,
    0.7069,
    0.5338,
    0.4889,
    0.4917,
    0.4069,
    0.4999,
    0.6866,
    0.4093,
    0.5709,
    0.6065,
    0.6415,
    0.4944,
    0.5726,
    1.2042,
    0.5458,
    1.6887,
    0.3971,
    1.06,
    0.3943,
    0.5537,
    0.5444,
    0.4089,
    0.7468,
    0.7744,
)


class _WanAutoencoder(nn.Module):
    def __init__(
        self,
        dim=160,
        dec_dim=256,
        z_dim=48,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_downsample=[False, True, True],
        dropout=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample
        self.temperal_upsample = temperal_downsample[::-1]

        # modules
        self.encoder = Encoder3d(
            dim,
            z_dim * 2,
            dim_mult,
            num_res_blocks,
            attn_scales,
            self.temperal_downsample,
            dropout,
        )
        self.conv1 = CausalConv3d(z_dim * 2, z_dim * 2, 1)
        self.conv2 = CausalConv3d(z_dim, z_dim, 1)
        self.decoder = Decoder3d(
            dec_dim,
            z_dim,
            dim_mult,
            num_res_blocks,
            attn_scales,
            self.temperal_upsample,
            dropout,
        )

        self._enc_conv_num = count_conv3d(self.encoder)
        self._dec_conv_num = count_conv3d(self.decoder)

    def encode(self, x, scale):
        """Prime the causal cache on frame zero, then encode four frames at a time."""
        cache = [None] * self._enc_conv_num
        x = patchify(x, patch_size=2)
        chunks = [x[:, :, :1], *x[:, :, 1:].split(4, dim=2)] if x.shape[2] > 1 else [x]
        outputs = []
        mean, inv_std = scale
        for chunk in chunks:
            features = self.encoder(chunk.contiguous(), feat_cache=cache)
            mu, _ = self.conv1(features).chunk(2, dim=1)
            outputs.append((mu - mean) * inv_std)
        return torch.cat(outputs, dim=2)

    def decode(self, z, scale):
        """Decode each latent frame with causal context from the preceding frames."""
        cache = [None] * self._dec_conv_num
        mean, inv_std = scale
        features = self.conv2(z / inv_std + mean)
        frames = [
            self.decoder(chunk, feat_cache=cache, first_chunk=(i == 0))
            for i, chunk in enumerate(features.split(1, dim=2))
        ]
        return unpatchify(torch.cat(frames, dim=2), patch_size=2)


class WanVideoVAE(nn.Module):
    """Frozen Wan2.2 4×16×16 video tokenizer with per-channel latent normalization."""

    spatial_compression_factor = 16
    temporal_compression_factor = 4
    latent_ch = 48

    def __init__(self, network, dtype, device):
        super().__init__()
        self.model = network
        mean = torch.tensor(_LATENT_MEAN, dtype=dtype, device=device)
        std = torch.tensor(_LATENT_STD, dtype=dtype, device=device)
        self.register_buffer("mean", mean.view(1, 48, 1, 1, 1), persistent=False)
        self.register_buffer("inv_std", (1.0 / std).view(1, 48, 1, 1, 1), persistent=False)

    @classmethod
    def from_checkpoint(cls, path, device="cuda", dtype=torch.bfloat16):
        """Load the official local Wan2.2_VAE.pth state dictionary."""
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Wan2.2 VAE weights not found: {path}")
        state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        with torch.device("meta"):
            network = _WanAutoencoder()
        network.load_state_dict(state, strict=True, assign=True)
        network.to(device=device, dtype=dtype)
        return cls(network, dtype=dtype, device=device).eval().requires_grad_(False)

    @property
    def dtype(self):
        return self.mean.dtype

    @staticmethod
    def get_latent_num_frames(num_pixel_frames):
        return 1 + (num_pixel_frames - 1) // 4

    @staticmethod
    def get_pixel_num_frames(num_latent_frames):
        return 1 + 4 * (num_latent_frames - 1)

    @torch.inference_mode()
    def encode(self, video):
        """Encode RGB videos [B, 3, 4k+1, H, W] in [-1, 1]."""
        if video.ndim != 5 or video.shape[1] != 3 or not video.is_floating_point():
            raise ValueError("video must be floating-point with shape [B, 3, T, H, W]")
        t, h, w = video.shape[2:]
        if (t - 1) % 4 or h % 16 or w % 16 or min(t, h, w) == 0:
            raise ValueError("Wan2.2 input requires 4k+1 frames and image dimensions divisible by 16")
        latent = self.model.encode(video.to(self.mean), (self.mean, self.inv_std))
        return latent.to(dtype=video.dtype)

    @torch.inference_mode()
    def decode(self, latent):
        """Decode [B, 48, T, H, W] to RGB frames in the input floating-point dtype."""
        if latent.ndim != 5 or latent.shape[1] != 48 or not latent.is_floating_point():
            raise ValueError("latent must be floating-point with shape [B, 48, T, H, W]")
        video = self.model.decode(latent.to(self.mean), (self.mean, self.inv_std))
        return video.to(dtype=latent.dtype)


load_vae = WanVideoVAE.from_checkpoint
