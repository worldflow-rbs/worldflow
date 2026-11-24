# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1 AND Apache-2.0
# WorldFlow retains the Qwen layers and rotary embedding functions used during inference.

# Source Repository: https://github.com/huggingface/transformers
# This is adapted from src/transformers/models/qwen3_vl/modeling_qwen3_vl.py.
# Commit Hash: 41e5abac5cb49983a08ddef3e8645d6efd23c8f3
"""Qwen3-VL feed-forward layers and interleaved rotary embeddings used in WorldFlow."""

from __future__ import annotations

import torch
from torch import nn
from transformers.activations import ACT2FN
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]  # [...,head_dim//2]
    x2 = x[..., x.shape[-1] // 2 :]  # [...,head_dim//2]
    return torch.cat((-x2, x1), dim=-1)  # [...,head_dim]


class Qwen3VLTextRotaryEmbedding(nn.Module):
    def __init__(self, config: Qwen3VLTextConfig):
        super().__init__()
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get("rope_type", "default")
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        self.mrope_section = (
            config.rope_scaling.get("mrope_section", [24, 20, 20]) if config.rope_scaling is not None else [24, 20, 20]
        )
        inv_freq, self.attention_scaling = self.rope_init_fn(self.config)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def init_weights(self, buffer_device: torch.device | None = None) -> None:
        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device=buffer_device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def apply_interleaved_mrope(self, freqs, mrope_section):
        """Apply interleaved MRoPE to 3D rotary embeddings.
        Reorganizes frequency layout from chunked [TTT...HHH...WWW] to
        interleaved [THTHWHTHW...TT], preserving frequency continuity.
        args:
            x: (3, bs, seq_len, head_dim // 2)
            mrope_section: (3,)
        returns:
            x_t: (bs, seq_len, head_dim // 2)
        """
        freqs_t = freqs[0]  # just overwrite the first dimension T
        for dim, offset in enumerate((1, 2), start=1):  # H, W
            length = mrope_section[dim] * 3
            idx = slice(offset, length, 3)
            freqs_t[..., idx] = freqs[dim, ..., idx]
        return freqs_t

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        assert self.inv_freq.dtype == torch.float32, f"inv_freq must be float32, but got {self.inv_freq.dtype}"
        assert position_ids.dtype in [torch.long, torch.float32], (
            f"position_ids must be long or float32, but got {position_ids.dtype}"
        )

        # In contrast to other models, Qwen3VL has different position ids for the grids
        # So we expand the inv_freq to shape (3, ...)
        if position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)  # [3,B,N]
        inv_freq_expanded = (
            self.inv_freq[None, None, :, None].expand(3, position_ids.shape[1], -1, 1).to(x.device)
        )  # [3,B,head_dim//2,1]
        position_ids_expanded = position_ids[:, :, None, :]  # [3,B,1,N]

        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)  # [3,B,N,head_dim//2]
        freqs = self.apply_interleaved_mrope(freqs, self.mrope_section)  # [B,N,head_dim//2]
        emb = torch.cat((freqs, freqs), dim=-1)  # [B,N,head_dim]
        cos = emb.cos() * self.attention_scaling  # [B,N,head_dim]
        sin = emb.sin() * self.attention_scaling  # [B,N,head_dim]

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)  # each: [B,N,head_dim]


class Qwen3VLTextRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        """
        Qwen3VLTextRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self) -> str:
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)  # [B,1,N,head_dim]
    sin = sin.unsqueeze(unsqueeze_dim)  # [B,1,N,head_dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)  # [B,num_heads,N,head_dim]
    k_embed = (k * cos) + (rotate_half(k) * sin)  # [B,num_kv_heads,N,head_dim]
    return q_embed, k_embed  # [B,num_heads,N,head_dim], [B,num_kv_heads,N,head_dim]


class Qwen3VLTextMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj
