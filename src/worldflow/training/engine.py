# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Optimizer-step training loop for WorldFlow's two released stages."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from .checkpoint import PowerEMA, checkpoint_info, restore_checkpoint, save_checkpoint
from .distributed import STAGE_SCALES, build_network, initialize_distributed


STAGE_STEPS = {1: 32_350, 2: 51_100}


@dataclass(frozen=True)
class TrainConfig:
    stage: int
    checkpoint: Path
    manifest: Path
    output: Path | None = None
    resume: Path | None = None
    steps: int | None = None
    stop_after: int | None = None
    batch_size: int = 2
    grad_accum: int = 1
    workers: int = 0
    seed: int = 42
    sampling: str = "curriculum"
    learning_rate: float = 1e-4
    warmup_steps: int = 500
    save_every: int = 2000
    log_every: int = 20
    resume_step: int | None = None

    @property
    def run_dir(self):
        return self.output if self.output is not None else self.checkpoint / "runs" / f"stage{self.stage}"

    @property
    def resume_dir(self):
        if self.resume_step is not None:
            return self.run_dir / f"iter_{self.resume_step:09d}"
        return self.resume

    @property
    def total_steps(self):
        return STAGE_STEPS[self.stage] if self.steps is None else self.steps

    def validate(self):
        if self.stage not in STAGE_STEPS or self.sampling not in ("curriculum", "uniform"):
            raise ValueError("Expected stage 1 or 2 and curriculum or uniform sampling")
        if self.resume_step is not None and (self.resume is not None or self.resume_step < 0):
            raise ValueError("Choose either --resume or a non-negative --resume-step")
        if min(self.total_steps, self.batch_size, self.grad_accum, self.save_every, self.log_every) <= 0:
            raise ValueError("Steps, batch size, accumulation and logging/checkpoint intervals must be positive")
        if self.stop_after is not None and not 0 < self.stop_after <= self.total_steps:
            raise ValueError("--stop-after must be between 1 and the full training horizon")
        if (
            min(self.workers, self.warmup_steps, self.seed) < 0
            or not math.isfinite(self.learning_rate)
            or self.learning_rate <= 0
        ):
            raise ValueError("Expected non-negative workers, seed and warmup, and a finite positive learning rate")
        warmup_end = self.total_steps * 11500 / 51100 if self.stage == 2 else int(self.total_steps * 1.2)
        if self.warmup_steps >= warmup_end:
            raise ValueError("Reduce --warmup-steps for this training horizon")


def lr_multiplier(stage, step, total, warmup):
    """Reference stage-one cosine or stage-two five-domain stage-aware schedule."""
    if step < warmup:
        return step / warmup
    if stage == 1:
        progress = min(1.0, (step - warmup) / (int(total * 1.2) - warmup))
        return (1 + math.cos(math.pi * progress)) / 2
    # Sim -> Molmo is the only early transition. The outdoor and early Real3
    # domain phases share the .8 plateau, followed by late Real3 and Demo5 decay.
    ratio = total / STAGE_STEPS[2]
    knots = (
        (warmup, 1.0),
        (11500 * ratio, 1.0),
        (12500 * ratio, 0.8),
        (42100 * ratio, 0.8),
        (48100 * ratio, 0.2),
        (total, 0.05),
    )
    for (begin, left), (end, right) in zip(knots, knots[1:]):
        if step < end:
            progress = (step - begin) / (end - begin)
            return left + (right - left) * (1 - math.cos(math.pi * progress)) / 2
    return knots[-1][1]


def _digest(paths):
    digest = hashlib.sha256()
    for path in paths:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(4 << 20), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _identity(config, world_size):
    hashes = [None]
    if dist.get_rank() == 0:
        tokenizer = config.checkpoint / "tokenizer"
        hashes[0] = {
            "manifest_sha256": _digest([config.manifest]),
            "tokenizer_sha256": _digest(sorted(path for path in tokenizer.iterdir() if path.is_file())),
            "vae_sha256": _digest([config.checkpoint / "Wan2.2_VAE.pth"]),
        }
    dist.broadcast_object_list(hashes, src=0)
    return {
        **hashes[0],
        "stage": config.stage,
        "precision": "bf16_live_bf16_adam_fp32_ema",
        "total_steps": config.total_steps,
        "batch_size": config.batch_size,
        "grad_accum": config.grad_accum,
        "world_size": world_size,
        "seed": config.seed,
        "sampling": config.sampling,
        "learning_rate": config.learning_rate,
        **({"workers": config.workers} if config.workers > 1 else {}),
        "warmup_steps": config.warmup_steps,
    }


def _log_metrics(metrics, step, lr, grad_norm):
    keys = sorted(metrics)
    values = torch.stack([metrics[key].detach().float() for key in keys])
    dist.all_reduce(values, op=dist.ReduceOp.AVG)
    if dist.get_rank() == 0:
        record = {"step": step, "lr": lr, "grad_norm": float(grad_norm)}
        record.update(zip(keys, values.tolist()))
        print(json.dumps(record), flush=True)


def train(config: TrainConfig):
    from transformers import AutoTokenizer
    from ..data import TrainingDataset, collate_training_samples
    from ..inference.loader import validate_shared_assets
    from ..models.vae import load_vae
    from .curriculum import TrainingBatchSampler
    from .objective import training_loss

    config.validate()
    validate_shared_assets(config.checkpoint)
    resume = config.resume_dir
    run_dir = config.run_dir
    if not config.manifest.is_file():
        raise FileNotFoundError(config.manifest)
    if resume is not None and not (resume / "training.json").is_file():
        raise FileNotFoundError(f"Missing WorldFlow training checkpoint at {resume}")
    if config.resume_step is not None and checkpoint_info(resume)["step"] != config.resume_step:
        raise ValueError(f"Saved iteration in {resume} differs from --resume-step {config.resume_step}")
    if resume is None and any(run_dir.glob("iter_*/training.json")):
        raise FileExistsError(f"Existing checkpoints in {run_dir}; resume an iteration or choose a new run directory")
    device, mesh = initialize_distributed()
    try:
        rank, size = dist.get_rank(), dist.get_world_size()
        torch.manual_seed(config.seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        identity = _identity(config, size)
        dataset = TrainingDataset(config.manifest)
        network = build_network(config.checkpoint, config.stage, device, mesh, resume=resume is not None)
        tokenizer = AutoTokenizer.from_pretrained(config.checkpoint / "tokenizer", local_files_only=True)
        vae = load_vae(config.checkpoint / "Wan2.2_VAE.pth", device=device, dtype=torch.bfloat16)
        vae.requires_grad_(False).eval()
        parameters = [p for p in network.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            parameters, lr=config.learning_rate, betas=(0.9, 0.95), eps=1e-6, weight_decay=0, fused=True
        )
        ema = PowerEMA(
            network,
            {
                "objectflow_scale_buf": STAGE_SCALES[config.stage],
                "objectflow_point_pe_abs_scale_buf": network.objectflow_point_pe.abs_scale,
                "objectflow_point_pe_rel_scale_buf": network.objectflow_point_pe.rel_scale,
            },
        )
        completed = 0
        if resume is not None:
            completed = restore_checkpoint(resume, network, ema, optimizer, identity)
            expected_scale = torch.tensor(STAGE_SCALES[config.stage], dtype=network.objectflow_scale_buf.dtype).item()
            if float(network.objectflow_scale_buf) != expected_scale:
                raise ValueError("Checkpoint flow scale differs from selected training stage")
        stop = config.total_steps if config.stop_after is None else config.stop_after
        if completed >= stop:
            raise ValueError(f"Checkpoint already completed {completed} steps; requested stop is {stop}")
        sampler = TrainingBatchSampler(
            dataset,
            stage=config.stage,
            batch_size=config.batch_size,
            total_steps=config.total_steps,
            grad_accum=config.grad_accum,
            rank=rank,
            world_size=size,
            seed=config.seed,
            start_step=completed,
            sampling=config.sampling,
        )
        loader = DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=collate_training_samples,
            num_workers=config.workers,
            pin_memory=True,
        )
        batches = iter(loader)
        if rank == 0:
            print(
                f"WorldFlow stage {config.stage}, step {completed} to {stop}, {size} ranks, "
                f"global batch {size * config.batch_size * config.grad_accum}",
                flush=True,
            )
        for step in range(completed, stop):
            lr = config.learning_rate * lr_multiplier(config.stage, step, config.total_steps, config.warmup_steps)
            optimizer.param_groups[0]["lr"] = lr
            metrics = {}
            for accumulation in range(config.grad_accum):
                samples = next(batches)
                microstep = step * config.grad_accum + accumulation
                # The reference dropout key uses the per-worker sample cursor.
                # Recreate the logical round-robin stream even after a restart.
                workers = max(1, config.workers)
                for index, sample in enumerate(samples):
                    sample["sample_worker_id"] = microstep % workers
                    sample["mixed_sample_position"] = (microstep // workers) * config.batch_size + index
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss, outputs = training_loss(
                        network,
                        vae,
                        tokenizer,
                        samples,
                        stage=config.stage,
                        seed=config.seed,
                        rank=rank,
                        noise_seed=config.seed + microstep * size + rank,
                    )
                (loss / config.grad_accum).backward()
                for key, value in outputs.items():
                    value = torch.as_tensor(value, device=device).detach() / config.grad_accum
                    metrics[key] = metrics.get(key, 0) + value
            for param in parameters:
                if param.grad is not None:
                    param.grad.to_local().nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
            norm = torch.nn.utils.clip_grad_norm_(parameters, 0.1, error_if_nonfinite=True, foreach=False)
            if hasattr(norm, "full_tensor"):
                norm = norm.full_tensor()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            ema.update(step)
            if (step + 1) % config.log_every == 0 or step == completed:
                _log_metrics(metrics, step + 1, lr, norm)
            if (step + 1) % config.save_every == 0 or step + 1 == stop:
                saved = save_checkpoint(run_dir, network, ema, optimizer, step + 1, identity)
                if rank == 0:
                    print(f"Saved {saved}", flush=True)
    finally:
        dist.destroy_process_group()
