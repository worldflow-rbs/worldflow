# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Position-based sampling for the SpatialCog warm-up and five-phase mixed-data recipe."""

from __future__ import annotations

import math
import random
from itertools import accumulate

import torch
from torch.utils.data import Sampler

from worldflow.inference.presets import SOURCE_DOMAINS


DOMAINS = ("sim", "molmo", "outdoor", "real3", "demo5")
PHASE_STEPS = (12000, 12000, 100, 24000, 3000)
_TRANSITIONS = (1000, 50, 50, 1000)
_MIX = (
    (0.80, 0.05, 0.00, 0.10, 0.05),
    (0.10, 0.70, 0.00, 0.15, 0.05),
    (0.05, 0.20, 0.50, 0.25, 0.00),
    (0.10, 0.10, 0.00, 0.75, 0.05),
    (0.05, 0.10, 0.00, 0.35, 0.50),
)


def domain_probabilities(step: int, total_steps: int) -> dict[str, float]:
    """Linearly blend domain probabilities around each reference phase boundary."""
    if step < 0 or total_steps <= 0:
        raise ValueError("Expected a non-negative step and positive training horizon")
    position = step * sum(PHASE_STEPS) / total_steps
    for phase, (boundary, width) in enumerate(zip(accumulate(PHASE_STEPS), _TRANSITIONS)):
        fraction = (position - boundary) / width + 0.5
        if fraction < 0:
            return dict(zip(DOMAINS, _MIX[phase], strict=True))
        if fraction < 1:
            weights = (a + (b - a) * fraction for a, b in zip(_MIX[phase], _MIX[phase + 1], strict=True))
            return dict(zip(DOMAINS, weights, strict=True))
    return dict(zip(DOMAINS, _MIX[-1], strict=True))


class TrainingBatchSampler(Sampler[list[int]]):
    """Shuffle Stage 1 by epoch and draw Stage 2 mixtures by optimizer step.

    The position is the optimizer step before the update. DataLoader prefetch cannot
    advance resumed training because it does not own the random stream.
    """

    def __init__(
        self,
        dataset,
        *,
        stage: int,
        batch_size: int,
        total_steps: int,
        grad_accum: int,
        rank: int,
        world_size: int,
        seed: int,
        start_step: int = 0,
        sampling: str = "curriculum",
    ):
        if stage not in (1, 2) or sampling not in ("curriculum", "uniform"):
            raise ValueError("Expected training stage 1 or 2 and sampling 'curriculum' or 'uniform'")
        if min(batch_size, total_steps, grad_accum, world_size, len(dataset)) <= 0:
            raise ValueError("Training requires a non-empty dataset and positive batch, step and process counts")
        if not 0 <= rank < world_size or not 0 <= start_step <= total_steps:
            raise ValueError("Invalid distributed rank or resume step")
        self.indices = dataset.source_indices
        self.count = len(dataset)
        self.sources = sorted(self.indices)
        self.sampling = sampling
        self.stage = stage
        self.batch_size = batch_size
        self.total_steps = total_steps
        self.grad_accum = grad_accum
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.start_step = start_step
        if sampling == "uniform":
            return
        if stage == 1:
            if self.sources != ["demo5"]:
                raise ValueError("The stage 1 recipe requires a Demo5-only manifest")
            if self.count < batch_size:
                raise ValueError("Stage 1 needs at least one full local batch")
            return
        unknown = set(self.sources) - SOURCE_DOMAINS.keys()
        if unknown:
            raise ValueError(f"Unrecognized stage 2 sources: {sorted(unknown)}")
        domains = {SOURCE_DOMAINS[s] for s in self.sources}
        if missing := set(DOMAINS) - domains:
            raise ValueError(f"Stage 2 curriculum requires all five domains. Missing {sorted(missing)}")
        weights = {s: len(self.indices[s]) ** (0.8 if SOURCE_DOMAINS[s] == "real3" else 0.5) for s in self.sources}
        totals = {d: math.fsum(w for s, w in weights.items() if SOURCE_DOMAINS[s] == d) for d in domains}
        self.shares = {s: w / totals[SOURCE_DOMAINS[s]] for s, w in weights.items()}

    def __len__(self):
        return (self.total_steps - self.start_step) * self.grad_accum

    def __iter__(self):
        if self.sampling == "curriculum" and self.stage == 1:
            # The warm-up loader shuffles the full Demo5 index on each rank
            # and drops the incomplete last batch of each local epoch.
            per_epoch = self.count // self.batch_size
            last_epoch = None
            for batch in range(self.start_step * self.grad_accum, self.total_steps * self.grad_accum):
                epoch, offset = divmod(batch, per_epoch)
                if last_epoch != epoch:
                    generator = torch.Generator().manual_seed(self.seed + epoch * self.world_size + self.rank)
                    permutation = torch.randperm(self.count, generator=generator)
                    last_epoch = epoch
                start = offset * self.batch_size
                yield permutation[start : start + self.batch_size].tolist()
            return
        for step in range(self.start_step, self.total_steps):
            if self.sampling == "curriculum" and self.stage == 2:
                mass = domain_probabilities(step, self.total_steps)
                cumulative = list(accumulate(mass[SOURCE_DOMAINS[s]] * self.shares[s] for s in self.sources))
            for microbatch in range(self.grad_accum):
                ordinal = (step * self.grad_accum + microbatch) * self.world_size + self.rank
                rng = random.Random(f"{self.seed}/{ordinal}")
                if self.sampling == "uniform":
                    yield [rng.randrange(self.count) for _ in range(self.batch_size)]
                else:
                    sources = rng.choices(self.sources, cum_weights=cumulative, k=self.batch_size)
                    yield [rng.choice(self.indices[s]) for s in sources]
