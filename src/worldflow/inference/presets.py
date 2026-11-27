# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Prompt contexts and metric scales for the published WorldFlow checkpoint."""

from collections.abc import Mapping

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
_CONTEXT_VALUES = {
    "origin": {"simulation", "real"},
    "actor": {"robot", "human", "mixed", "none"},
    "view": {"ego", "external_fixed", "external_moving"},
    "scene": {"indoor", "outdoor", "mixed"},
}
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


def flow_scale(source: str, stage: int = 2) -> float:
    """Meters per normalized unit of frame-to-frame motion."""
    if stage == 1:
        return 0.0155919
    if stage != 2:
        raise ValueError("stage must be 1 or 2")
    if source not in SOURCE_DOMAINS:
        raise ValueError(f"unknown source {source!r}")
    return DOMAIN_SCALES[SOURCE_DOMAINS[source]]


def format_prompt(instruction: str, source: str, prompt_context: Mapping | None = None) -> str:
    """Format a stage-two prompt using per-example context or the source defaults."""
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("instruction must be a non-empty string")
    if source not in _PRESETS:
        raise ValueError(f"unknown source {source!r}; choose one of {', '.join(_PRESETS)}")
    values = _PRESETS[source][1:]
    if prompt_context is not None:
        if not isinstance(prompt_context, Mapping) or set(prompt_context) != set(_CONTEXT_FIELDS):
            raise ValueError(f"prompt_context must contain exactly {_CONTEXT_FIELDS}")
        values = tuple(str(prompt_context[key]).strip().lower() for key in _CONTEXT_FIELDS)
        for key, value in zip(_CONTEXT_FIELDS, values, strict=True):
            if value not in _CONTEXT_VALUES[key]:
                raise ValueError(f"invalid prompt_context {key}={value!r}")
    context = "\n".join(f"{key}={value}" for key, value in zip(_CONTEXT_FIELDS, values, strict=True))
    return f"[INSTRUCTION]\n{instruction.strip()}\n[/INSTRUCTION]\n\n[CONTEXT]\n{context}\n[/CONTEXT]"


def model_prompt(instruction: str, source: str, *, stage: int = 2, prompt_context: Mapping | None = None) -> str:
    if stage == 1:
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("instruction must be a non-empty string")
        return instruction
    if stage != 2:
        raise ValueError("stage must be 1 or 2")
    return format_prompt(instruction, source, prompt_context)
