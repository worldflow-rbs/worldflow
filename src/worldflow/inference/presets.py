# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Prompt contexts and metric scales for the published WorldFlow checkpoint."""

DOMAIN_SCALES = {
    "demo5": 0.01934810744226477,
    "sim": 0.010821323375216966,
    "real3": 0.010226523439128974,
    "molmo": 0.022936180674055843,
    "outdoor": 0.3219299307295263,
}

# Domain determines the displacement scale. The other four fields comprise the
# SPV1 context given to the model; source names never enter the prompt.
_CONTEXT_FIELDS = ("origin", "actor", "view", "scene")
_PRESETS = {
    "sim7": ("sim", "simulation", "robot", "external_fixed", "indoor"),
    "demo5": ("demo5", "real", "robot", "external_fixed", "indoor"),
    "agibot": ("real3", "real", "robot", "ego", "indoor"),
    "sthv2": ("real3", "real", "human", "external_fixed", "indoor"),
    "epic": ("real3", "real", "human", "ego", "indoor"),
    "molmo_droid": ("molmo", "real", "robot", "external_fixed", "indoor"),
    "molmo_egodex": ("molmo", "real", "human", "ego", "indoor"),
    "molmo_molmospaces": ("sim", "simulation", "robot", "external_fixed", "indoor"),
    "molmo_stereo4d": ("molmo", "real", "human", "external_moving", "indoor"),
    "molmo_ytvis": ("outdoor", "real", "mixed", "external_moving", "outdoor"),
    "molmo_xperience": ("molmo", "real", "human", "external_fixed", "indoor"),
}
SOURCE_DOMAINS = {source: preset[0] for source, preset in _PRESETS.items()}


def format_prompt(instruction: str, source: str) -> str:
    """Format a plain-language instruction exactly as in the SPV1 evaluation."""
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("instruction must be a non-empty string")
    if source not in _PRESETS:
        raise ValueError(f"unknown source {source!r}; choose one of {', '.join(_PRESETS)}")
    values = _PRESETS[source][1:]
    context = "\n".join(f"{key}={value}" for key, value in zip(_CONTEXT_FIELDS, values, strict=True))
    return f"[INSTRUCTION]\n{instruction.strip()}\n[/INSTRUCTION]\n\n[CONTEXT]\n{context}\n[/CONTEXT]"
