# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Single-observation joint video and object-flow sampling for WorldFlow Nano."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from ..models.packing import pack_joint
from .scheduler import FlowUniPCMultistepScheduler


def _initial_noise(shape, seed, device):
    # The reference initializes each modality with an independent RandomState
    # using the same sample seed. Changing RNGs changes both predictions.
    noise = np.random.RandomState(seed).standard_normal(shape).astype(np.float32)
    return torch.from_numpy(noise).to(device)


class JointSampler(nn.Module):
    """Denoise a video and its 64 query trajectories in a single sequence."""

    def __init__(self, net, vae, tokenizer):
        super().__init__()
        self.net = net
        self.vae = vae
        self.tokenizer = tokenizer
        self.eos_id = tokenizer.eos_token_id
        self.vision_start_id = tokenizer.convert_tokens_to_ids("<|vision_start|>")
        if self.eos_id is None or self.vision_start_id in (None, tokenizer.unk_token_id):
            raise ValueError("The checkpoint tokenizer must contain EOS and <|vision_start|> tokens")

    @property
    def device(self):
        return next(self.net.parameters()).device

    def decode(self, latent):
        return self.vae.decode(latent)

    def _caption_ids(self, prompt):
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
            add_vision_id=False,
            return_dict=False,
        )

    def _velocity(self, state, timestep, ids, batch, vision_shape, flow_shape):
        n_vision = vision_shape.numel()
        vision = state[:n_vision].reshape(vision_shape)
        flow = state[n_vision:].reshape(flow_shape)
        packed = pack_joint(
            ids,
            vision,
            flow,
            timestep,
            batch["points_xyz"],
            batch["pixel_indices"],
            eos_id=self.eos_id,
            vision_start_id=self.vision_start_id,
        )
        out = self.net(packed_seq=packed)
        # The conditioned frame is fixed throughout sampling. All flow slots
        # are generated and therefore carry unit velocity masks.
        pred_video = out["preds_vision"][0]
        mask = (1.0 - packed.vision.condition_mask[0]).to(pred_video)
        velocity_video = pred_video * mask
        return torch.cat((velocity_video.flatten(), out["preds_objectflow"][0].flatten()))

    @torch.inference_mode()
    def generate_samples_from_batch(self, batch, *, seed, num_steps, guidance, shift):
        frame = batch["frame"][None, :, None].float() / 127.5 - 1.0
        device = frame.device
        # Temporal causality makes the encoded first frame identical to the
        # prefix of the full clip. Every later latent frame starts from noise.
        anchor = self.vae.encode(frame).contiguous().float()
        latent_frames = self.vae.get_latent_num_frames(batch["num_frames"])
        vision_shape = torch.Size((*anchor.shape[:2], latent_frames, *anchor.shape[3:]))
        flow_shape = torch.Size((batch["num_frames"] - 1, *batch["points_xyz"].shape))
        vision_noise = _initial_noise(vision_shape, seed[0], device)
        vision_noise[:, :, :1] = anchor
        flow_noise = _initial_noise(flow_shape, seed[0], device)
        state = torch.cat((vision_noise.flatten(), flow_noise.flatten()))
        cond_ids = self._caption_ids(batch["prompt"])
        uncond_ids = self._caption_ids("")

        scheduler = FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=1.0, use_dynamic_shifting=False)
        scheduler.set_timesteps(num_steps, device=device, shift=float(shift))
        generator = torch.Generator(device=device).manual_seed(seed[0])
        for timestep in scheduler.timesteps:
            sigma = timestep.reshape(1, 1)
            velocity = self._velocity(state, sigma, cond_ids, batch, vision_shape, flow_shape)
            if guidance != 1.0:
                uncond = self._velocity(state, sigma, uncond_ids, batch, vision_shape, flow_shape)
                velocity = uncond + guidance * (velocity - uncond)
            next_state = scheduler.step(
                model_output=velocity,
                timestep=timestep,
                sample=state.unsqueeze(0),
                return_dict=False,
                generator=generator,
            )
            state = next_state[0].squeeze(0)
        split = vision_shape.numel()
        return {
            "vision": [state[:split].reshape(vision_shape)],
            "objectflow": [state[split:].reshape(flow_shape)],
        }
