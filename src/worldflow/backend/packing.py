# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Pack one text prompt, video latent and set of query trajectories for WorldFlow."""

from dataclasses import dataclass

import torch


@dataclass
class ModalityTokens:
    tokens: list[torch.Tensor]
    token_shapes: list[tuple[int, int, int]]
    sequence_indexes: torch.Tensor
    mse_loss_indexes: torch.Tensor
    timesteps: torch.Tensor
    noisy_frame_indexes: list[torch.Tensor]
    condition_mask: list[torch.Tensor]


@dataclass
class PackedSequence:
    text_ids: torch.Tensor
    text_indexes: torch.Tensor
    position_ids: torch.Tensor
    sequence_length: int
    sample_lens: list[int]
    split_lens: list[int]
    attn_modes: list[str]
    vision: ModalityTokens
    objectflow: ModalityTokens
    objectflow_points_src: list[torch.Tensor]


def pack_joint(
    text_ids, vision, flow, timestep, points_xyz, pixel_indices, *, eos_id, vision_start_id, fps=16.0
) -> PackedSequence:
    """Build the checkpoint's causal text prefix and full-attention generation span.

    Video latents are [1, C, T, H, W]. Flow is [F, N, 3]. Pixel indices are
    row-major in the input image, and align with the rows of points_xyz.
    Only the first video latent is conditioned. All flow tokens are generated.
    """
    device = vision.device
    _, _, frames, height, width = vision.shape
    steps, points, _ = flow.shape
    rows, cols = (height + 1) // 2, (width + 1) // 2
    patches = rows * cols
    ids = torch.as_tensor(text_ids, device=device, dtype=torch.long)
    ids = torch.cat((ids, ids.new_tensor([eos_id, vision_start_id])))
    text_length = ids.numel()
    vision_end = text_length + frames * patches
    length = vision_end + steps * points
    indices = torch.arange(length, device=device)
    time_origin = text_length + 15000

    # The video grid uses the 24 fps reference clock. Each flow step uses a
    # quarter-latent time unit from frame 1; its spatial coordinates stay on
    # the patch containing the point's observed source pixel.
    frame_numbers = torch.arange(frames, dtype=torch.float32, device=device)
    video_time = frame_numbers / (fps / 4) * 6 + time_origin
    video_positions = torch.stack(
        torch.meshgrid(
            video_time,
            torch.arange(rows, device=device, dtype=torch.float32),
            torch.arange(cols, device=device, dtype=torch.float32),
            indexing="ij",
        )
    ).reshape(3, -1)
    flow_frames = torch.arange(steps, device=device)
    flow_time = (flow_frames.float() + 1) / 4 + time_origin
    image_width = width * 16
    pixels = pixel_indices.to(device=device, dtype=torch.long)
    flow_positions = torch.stack(
        (
            flow_time.repeat_interleave(points),
            ((pixels // image_width) // 32).repeat(steps),
            ((pixels % image_width) // 32).repeat(steps),
        )
    )
    text_positions = indices[:text_length].float().expand(3, -1)
    positions = torch.cat((text_positions, video_positions, flow_positions), dim=1)

    noisy_video = indices[text_length + patches : vision_end]
    motion = indices[vision_end:]
    sigma = torch.as_tensor(timestep, device=device, dtype=torch.float32).reshape(())
    video_mask = vision.new_zeros((frames, 1, 1))
    video_mask[0] = 1
    video_tokens = ModalityTokens(
        tokens=[vision.to(torch.bfloat16)],
        token_shapes=[(frames, rows, cols)],
        sequence_indexes=indices[text_length:vision_end],
        mse_loss_indexes=noisy_video,
        timesteps=sigma.expand(noisy_video.numel()),
        noisy_frame_indexes=[indices[1:frames]],
        condition_mask=[video_mask],
    )
    flow_tokens = ModalityTokens(
        tokens=[flow.to(torch.bfloat16)],
        token_shapes=[(steps, points, 1)],
        sequence_indexes=motion,
        mse_loss_indexes=motion,
        timesteps=sigma.expand(motion.numel()),
        noisy_frame_indexes=[flow_frames],
        condition_mask=[flow.new_zeros((steps, 1))],
    )
    return PackedSequence(
        text_ids=ids,
        text_indexes=indices[:text_length],
        position_ids=positions,
        sequence_length=length,
        sample_lens=[length],
        split_lens=[text_length, length - text_length],
        attn_modes=["causal", "full"],
        vision=video_tokens,
        objectflow=flow_tokens,
        objectflow_points_src=[points_xyz.to(device)],
    )
