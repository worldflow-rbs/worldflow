# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Sharded training checkpoints with a portable WorldFlow EMA model."""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.tensor import DTensor
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    get_optimizer_state_dict,
    set_optimizer_state_dict,
)


# Power-function EMA (sigma_rel=0.1) used by both stages of the released model.
EMA_EXPONENT = float(np.roots([1, 7, 16 - 0.1**-2, 12 - 0.1**-2]).real.max()) + 1


class PowerEMA:
    """Power-average all parameter shards in FP32, including the frozen language path."""

    def __init__(self, network, buffer_values=None):
        self.network = network
        state = get_model_state_dict(network)
        self.shadow = {
            name: value.detach().to(dtype=torch.float32, copy=True)
            if value.is_floating_point()
            else value.detach().clone()
            for name, value in state.items()
        }
        # The original EMA has its own FP32 buffers. Its power update visits
        # parameters only, so these config constants retain their full precision.
        for name, value in (buffer_values or {}).items():
            self.shadow[name].fill_(value)
        self.parameter_names = [
            name.replace("_checkpoint_wrapped_module.", "") for name, _ in network.named_parameters()
        ]

    @torch.no_grad()
    def update(self, iteration):
        # The reference uses the zero-based index of the optimizer update.
        beta = (iteration / (iteration + 1)) ** EMA_EXPONENT
        state = self.network.state_dict()
        targets, sources = [], []
        for name in self.parameter_names:
            target, source = self.shadow[name], state[name]
            targets.append(target.to_local() if isinstance(target, DTensor) else target)
            sources.append(source.to_local() if isinstance(source, DTensor) else source)
        torch._foreach_mul_(targets, beta)
        torch._foreach_add_(targets, sources, alpha=1 - beta)

    def model_state(self):
        state = get_model_state_dict(self.network)
        result = {f"net.{name}": value for name, value in state.items()}
        result.update({f"net_ema.{name}": value for name, value in self.shadow.items()})
        return result


def checkpoint_info(path):
    info = json.loads((Path(path) / "training.json").read_text(encoding="utf-8"))
    if info.get("format") != 1 or type(info.get("step")) is not int or info["step"] < 0:
        raise ValueError(f"Invalid WorldFlow training checkpoint metadata in {path}")
    return info


def _rank_zero():
    return not dist.is_initialized() or dist.get_rank() == 0


def save_checkpoint(root, network, ema, optimizer, step, identity):
    """Publish a complete model/Adam checkpoint with a single directory rename."""
    root = Path(root)
    destination = root / f"iter_{step:09d}"
    if destination.exists():
        raise FileExistsError(destination)
    # Unique sibling staging makes interrupted writes invisible to resume/export.
    token = [uuid.uuid4().hex if _rank_zero() else None]
    if dist.is_initialized():
        dist.broadcast_object_list(token, src=0)
    staging = root / f".{destination.name}.{token[0]}"
    staging.mkdir(parents=True, exist_ok=True)
    dcp.save(ema.model_state(), checkpoint_id=staging / "model", no_dist=not dist.is_initialized())
    dcp.save(
        get_optimizer_state_dict(network, optimizer),
        checkpoint_id=staging / "optimizer",
        no_dist=not dist.is_initialized(),
    )
    if _rank_zero():
        metadata = {"format": 1, "step": step, "identity": identity}
        (staging / "training.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        os.rename(staging, destination)
    if dist.is_initialized():
        dist.barrier()
    return destination


def restore_checkpoint(path, network, ema, optimizer, identity):
    """Restore the live network, power EMA, Adam moments and optimization cursor."""
    path = Path(path)
    info = checkpoint_info(path)
    if info.get("identity") != identity:
        previous = info.get("identity", {})
        changed = sorted(key for key in previous.keys() | identity.keys() if previous.get(key) != identity.get(key))
        raise ValueError(f"Resume settings differ from saved training run: {', '.join(changed)}")
    model_state = ema.model_state()
    dcp.load(model_state, checkpoint_id=path / "model", no_dist=not dist.is_initialized())
    # DCP fills tensor storage in-place. Only the optimizer also contains Python
    # objects (parameter groups, scalar hyperparameters) to bind after loading.
    opt_state = get_optimizer_state_dict(network, optimizer)
    dcp.load(opt_state, checkpoint_id=path / "optimizer", no_dist=not dist.is_initialized())
    set_optimizer_state_dict(network, optimizer, opt_state)
    return info["step"]
