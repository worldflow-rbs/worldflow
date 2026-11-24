# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Dense Cosmos3-Nano mixture of transformers for joint text, video and flow."""

import torch
from torch import nn
from torch.nn.functional import scaled_dot_product_attention
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig

from .qwen import (
    Qwen3VLTextMLP,
    Qwen3VLTextRMSNorm,
    Qwen3VLTextRotaryEmbedding,
    apply_rotary_pos_emb,
)


def _attention(query, key, value, causal=False):
    """Run grouped-query attention on one sequence in [tokens, heads, dim] layout."""

    def heads_first(tensor):
        return tensor.transpose(0, 1).unsqueeze(0)

    attended = scaled_dot_product_attention(
        heads_first(query),
        heads_first(key),
        heads_first(value),
        is_causal=causal,
        enable_gqa=True,
    )
    return attended.squeeze(0).transpose(0, 1).flatten(1)


class DualAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        dim = config.hidden_size
        q_dim = self.num_heads * self.head_dim
        kv_dim = self.num_kv_heads * self.head_dim
        for suffix in ("", "_moe_gen"):
            setattr(self, "q_proj" + suffix, nn.Linear(dim, q_dim, bias=False))
            setattr(self, "k_proj" + suffix, nn.Linear(dim, kv_dim, bias=False))
            setattr(self, "v_proj" + suffix, nn.Linear(dim, kv_dim, bias=False))
            setattr(self, "o_proj" + suffix, nn.Linear(q_dim, dim, bias=False))
            setattr(self, "q_norm" + suffix, Qwen3VLTextRMSNorm(self.head_dim, config.rms_norm_eps))
            setattr(self, "k_norm" + suffix, Qwen3VLTextRMSNorm(self.head_dim, config.rms_norm_eps))

    def _project(self, hidden, rotary, suffix):
        query = getattr(self, "q_proj" + suffix)(hidden).view(-1, self.num_heads, self.head_dim)
        key = getattr(self, "k_proj" + suffix)(hidden).view(-1, self.num_kv_heads, self.head_dim)
        value = getattr(self, "v_proj" + suffix)(hidden).view(-1, self.num_kv_heads, self.head_dim)
        query = getattr(self, "q_norm" + suffix)(query)
        key = getattr(self, "k_norm" + suffix)(key)
        query, key = apply_rotary_pos_emb(query, key, *rotary, unsqueeze_dim=1)
        return query, key, value

    def forward(self, text, generation, text_rope, generation_rope):
        text_q, text_k, text_v = self._project(text, text_rope, "")
        gen_q, gen_k, gen_v = self._project(generation, generation_rope, "_moe_gen")
        text = self.o_proj(_attention(text_q, text_k, text_v, causal=True))
        generation = self.o_proj_moe_gen(_attention(gen_q, torch.cat((text_k, gen_k)), torch.cat((text_v, gen_v))))
        return text, generation


class DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = DualAttention(config)
        self.mlp = Qwen3VLTextMLP(config)
        self.mlp_moe_gen = Qwen3VLTextMLP(config)
        for suffix in ("", "_moe_gen"):
            for name in ("input_layernorm", "post_attention_layernorm"):
                setattr(self, name + suffix, Qwen3VLTextRMSNorm(config.hidden_size, config.rms_norm_eps))

    def forward(self, text, generation, text_rope, generation_rope):
        text_out, gen_out = self.self_attn(
            self.input_layernorm(text), self.input_layernorm_moe_gen(generation), text_rope, generation_rope
        )
        text = text + text_out
        generation = generation + gen_out
        text = text + self.mlp(self.post_attention_layernorm(text))
        generation = generation + self.mlp_moe_gen(self.post_attention_layernorm_moe_gen(generation))
        return text, generation


class TextModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(DecoderLayer(config) for _ in range(config.num_hidden_layers))
        self.norm = Qwen3VLTextRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.norm_moe_gen = Qwen3VLTextRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rotary_emb = Qwen3VLTextRotaryEmbedding(config)

    def forward(self, embeddings, position_ids, text_length):
        cos, sin = self.rotary_emb(embeddings, position_ids.unsqueeze(1))
        cos, sin = cos.squeeze(0), sin.squeeze(0)
        text, generation = embeddings[:text_length], embeddings[text_length:]
        text_rope = (cos[:text_length], sin[:text_length])
        generation_rope = (cos[text_length:], sin[text_length:])
        for layer in self.layers:
            text, generation = layer(text, generation, text_rope, generation_rope)
        return torch.cat((self.norm(text), self.norm_moe_gen(generation)))


class LanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        config = Qwen3VLTextConfig(
            vocab_size=151936,
            hidden_size=4096,
            intermediate_size=12288,
            num_hidden_layers=36,
            num_attention_heads=32,
            num_key_value_heads=8,
            head_dim=128,
            hidden_act="silu",
            max_position_embeddings=262144,
            rms_norm_eps=1e-6,
            rope_theta=5_000_000,
            rope_scaling={"rope_type": "default", "mrope_interleaved": True, "mrope_section": [24, 20, 20]},
            attention_bias=False,
            attention_dropout=0.0,
        )
        self.model = TextModel(config)
        # The EMA checkpoint stores the language head in addition to the input embedding.
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
