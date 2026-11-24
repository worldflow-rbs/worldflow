# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Joint rectified-flow and trajectory objectives used in both WorldFlow training stages."""

from __future__ import annotations

import hashlib
import json
import math
import random

import torch
from torch.nn import functional as F

from ..inference.presets import flow_scale, model_prompt
from ..models.packing import pack_joint
from .augmentation import augment_demo5


def _training_prompt(sample, stage, probability, seed, rank):
    instruction = sample["instruction"]
    variants = sample.get("instruction_variants") or []
    cursor = [sample.get("sample_worker_id"), sample.get("mixed_sample_position")]
    if variants or (stage == 1 and sample["source"] == "demo5"):
        # Per-visit randomness remains reproducible when resuming or prefetching.
        rng = random.Random(json.dumps(["paraphrase", seed, rank, sample.get("sample_id"), *cursor]))
        if variants:
            instruction = rng.choice([instruction, *variants])
        if stage == 1 and sample["source"] == "demo5":
            instruction = augment_demo5(instruction, rng)
    prompt = model_prompt(
        instruction, sample["source"], stage=stage, prompt_context=sample.get("prompt_context")
    )
    if probability:
        if any(type(value) is not int or value < 0 for value in cursor):
            raise ValueError("CFG dropout requires non-negative sample_worker_id and mixed_sample_position")
        key = json.dumps(["all-text-cfg-v1", int(seed), int(rank), *cursor]).encode()
        draw = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") / 2**64
        if draw < probability:
            return ""
    return prompt


def _waver_sigmas(count, device, generator):
    # The distribution is sampled on CPU before the resolution-dependent shift.
    u = torch.rand((count,), dtype=torch.float32, device="cpu", generator=generator)
    time = (1.0 - u - 1.29 * (torch.cos(torch.pi / 2.0 * u) ** 2 - 1 + u)).to(device)
    return 5.0 * time / (1.0 + 4.0 * time)


def _velocity_loss(pred, target, mask):
    return ((pred.float() - target).square() * mask).mean()


def trajectory_losses(noisy, sigma, predicted_velocity, target_velocity, endpoint_floor):
    """Recover normalized step offsets from velocity and reduce over frames, points and samples.

    Inputs have shape [B, frames, points, 3], except sigma [B]. Weight heading by
    cumulative travel, suppress static endpoints, and give lower-noise samples more weight.
    """
    sigma = sigma.float()[:, None, None, None]
    pred_steps = noisy.float() - sigma * predicted_velocity.float()
    true_steps = noisy.float() - sigma * target_velocity.float()
    pred_offsets = pred_steps.cumsum(dim=1)
    true_offsets = true_steps.cumsum(dim=1)
    weight = (1.0 - sigma).squeeze(-1)
    eps = 1e-6

    cosine = F.cosine_similarity(pred_offsets, true_offsets, dim=-1, eps=eps)
    dir_weight = true_offsets.norm(dim=-1) * weight
    direction = ((1.0 - cosine) * dir_weight).sum() / (dir_weight.sum() + eps)

    step_error = (pred_steps - true_steps).square().sum(dim=-1)
    step_weight = weight.expand_as(step_error)
    velocity = (step_error * step_weight).sum() / (step_weight.sum() + eps)

    length = true_offsets[:, -1].norm(dim=-1)
    relative_error = (pred_offsets[:, -1] - true_offsets[:, -1]).norm(dim=-1) / length.clamp_min(endpoint_floor)
    end_weight = (length >= endpoint_floor).float() * weight[:, 0]
    endpoint = (relative_error * end_weight).sum() / (end_weight.sum() + eps)
    return direction, velocity, endpoint


def training_loss(
    network, vae, tokenizer, samples, *, stage=2, seed=42, rank=0, noise_seed=None,
    cfg_dropout=None, direction_weight=None, velocity_weight=None, endpoint_weight=None, endpoint_floor=0.1,
):
    """Return a differentiable loss and detached scalar metrics for a local microbatch.

    The VAE encodes all 49 frames under no_grad. The first encoded frame and
    the initial point geometry condition the denoiser. Each clip has one noise
    level shared across the 12 generated video latents and 48 flow steps.
    """
    if not samples:
        raise ValueError("training_loss requires at least one sample")
    if stage not in (1, 2):
        raise ValueError("stage must be 1 or 2")
    defaults = (0.0, 0.15, 0.015, 0.02) if stage == 1 else (0.1, 0.1, 0.1, 0.05)
    if cfg_dropout is None:
        cfg_dropout = defaults[0]
    if not 0 <= cfg_dropout <= 1:
        raise ValueError("cfg_dropout must be in [0, 1]")
    weights = tuple(
        default if value is None else value
        for value, default in zip((direction_weight, velocity_weight, endpoint_weight), defaults[1:], strict=True)
    )
    if any(not math.isfinite(w) or w < 0 for w in weights) or not math.isfinite(endpoint_floor) or endpoint_floor <= 0:
        raise ValueError("auxiliary weights must be non-negative and endpoint_floor must be positive")

    device = next(network.parameters()).device
    time_rng = noise_rng = None
    if noise_seed is not None:
        time_rng = torch.Generator(device="cpu").manual_seed(noise_seed)
        noise_rng = torch.Generator(device=device).manual_seed(noise_seed + 32768)
    sigmas = _waver_sigmas(len(samples), device, time_rng)
    eos = tokenizer.eos_token_id
    vision_start = tokenizer.convert_tokens_to_ids("<|vision_start|>")

    # Draw all video noise before all flow noise in the same order as the two
    # modality streams in the training checkpoint.
    latents, vision_noise = [], []
    with torch.no_grad():
        for sample in samples:
            pixels = torch.as_tensor(sample["video"]).permute(3, 0, 1, 2)[None]
            clip = pixels.to(device=device, dtype=torch.float32) / 127.5 - 1.0
            latent = vae.encode(clip).contiguous().float()
            latents.append(latent)
            vision_noise.append(torch.randn(latent.shape, device=device, dtype=torch.float32, generator=noise_rng))

    video_mse, flow_mse = [], []
    flow_inputs, flow_preds, flow_targets = [], [], []
    dropped = 0
    for sample, sigma, latent, eps_video in zip(samples, sigmas, latents, vision_noise, strict=True):
        clean_flow = torch.as_tensor(sample["flow_m"], device=device, dtype=torch.float32) / flow_scale(
            sample["source"], stage
        )
        eps_flow = torch.randn(clean_flow.shape, device=device, dtype=torch.float32, generator=noise_rng)
        active = torch.ones((latent.shape[2], 1, 1), device=device, dtype=torch.float32)
        active[0] = 0
        video_sigma = sigma * active
        noised_video = eps_video * video_sigma + latent * (1 - video_sigma)
        noised_flow = (eps_flow * sigma + clean_flow * (1 - sigma)).to(torch.bfloat16)
        prompt = _training_prompt(sample, stage, cfg_dropout, seed, rank)
        dropped += int(prompt == "")
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True, add_generation_prompt=True, add_vision_id=False, return_dict=False,
        )
        packed = pack_joint(
            ids, noised_video, noised_flow, sigma * 1000,
            torch.as_tensor(sample["points_xyz"], device=device),
            torch.as_tensor(sample["pixel_indices"], device=device),
            eos_id=eos, vision_start_id=vision_start,
        )
        output = network(packed_seq=packed)
        pred_flow = output["preds_objectflow"][0]
        target_flow = eps_flow - clean_flow
        video_mse.append(_velocity_loss(output["preds_vision"][0], eps_video - latent, active))
        flow_mse.append(_velocity_loss(pred_flow, target_flow, 1.0))
        flow_inputs.append(noised_flow)
        flow_preds.append(pred_flow)
        flow_targets.append(target_flow)

    vision = torch.stack(video_mse).mean()
    flow = torch.stack(flow_mse).mean()
    aux = trajectory_losses(
        torch.stack(flow_inputs), sigmas, torch.stack(flow_preds), torch.stack(flow_targets), endpoint_floor
    )
    total = vision + flow
    aux_total = total.new_zeros(())
    for value, weight in zip(aux, weights, strict=True):
        if weight > 0:
            aux_total = aux_total + weight * value
    total = total + aux_total
    metrics = {
        "loss": total.detach(),
        "video_fm_loss": vision.detach(),
        "flow_fm_loss": flow.detach(),
        "direction_loss": aux[0].detach(),
        "velocity_loss": aux[1].detach(),
        "endpoint_loss": aux[2].detach(),
        "text_drop_rate": total.new_tensor(dropped / len(samples)),
    }
    return total, metrics
