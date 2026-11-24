# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Projection and output heads for the joint WorldFlow denoiser."""

import math
from typing import List, Tuple

import torch
from torch import nn

from .packing import PackedSequence
from .geometry import ObjectFlowPointPE
from .timestep import TimestepEmbedder
from .transformer import LanguageModel


class WorldFlowNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.language_model = LanguageModel()
        self.hidden_size = self.language_model.model.embed_tokens.embedding_dim
        self.latent_patch_size = 2
        self.latent_channel = 48
        self.objectflow_dim = 3
        self.timestep_scale = 0.001
        self.patch_latent_dim = self.latent_patch_size**2 * self.latent_channel
        self.time_embedder = TimestepEmbedder(self.hidden_size)
        self.vae2llm = nn.Linear(self.patch_latent_dim, self.hidden_size)
        self.llm2vae = nn.Linear(self.hidden_size, self.patch_latent_dim)
        self.objectflow2llm = nn.Linear(self.objectflow_dim, self.hidden_size)
        self.llm2objectflow = nn.Linear(self.hidden_size, self.objectflow_dim)
        self.objectflow_modality_embed = nn.Parameter(torch.zeros(self.hidden_size))
        self.objectflow_point_pe = ObjectFlowPointPE(self.hidden_size)
        self.register_buffer("objectflow_scale_buf", torch.tensor(0.01934810744226477))
        self.register_buffer("objectflow_point_pe_abs_scale_buf", torch.tensor(2.0))
        self.register_buffer("objectflow_point_pe_rel_scale_buf", torch.tensor(0.3))

    def forward(self, packed_seq):
        if packed_seq.attn_modes != ["causal", "full"] or len(packed_seq.sample_lens) != 1:
            raise ValueError("Expected one sample with a causal text prefix and joint video/flow tokens")
        sequence, dtype = self._encode_text(packed_seq)
        original_shapes = self._encode_vision(packed_seq, sequence, dtype)
        self._encode_objectflow(packed_seq, sequence, dtype)
        hidden = self.language_model.model(sequence, packed_seq.position_ids, packed_seq.split_lens[0])

        vision = packed_seq.vision
        motion = packed_seq.objectflow
        predictions = self.llm2vae(hidden[vision.mse_loss_indexes])
        flow = self.llm2objectflow(hidden[motion.mse_loss_indexes])
        return {
            "preds_vision": self.unpatchify_and_unpack_latents(
                predictions, vision.token_shapes, vision.noisy_frame_indexes, original_shapes
            ),
            "preds_objectflow": self.unpack_objectflow_latents(flow, motion.token_shapes, motion.noisy_frame_indexes),
        }

    def patchify_and_pack_latents(
        self, tokens_vision: torch.Tensor, token_shapes_vision: List[Tuple[int, int, int]]
    ) -> tuple[torch.Tensor, List[Tuple[int, int, int]]]:
        p = self.latent_patch_size
        # Patchify and pack the latents
        packed_latent = []
        original_latent_shapes = []  # Store original shapes for unpadding later

        # C, T, H, W
        for latent, (t, h, w) in zip(tokens_vision, token_shapes_vision):
            latent = latent.squeeze(0)  # [C,T,H,W]

            # Get original latent dimensions
            _, t_actual, h_actual, w_actual = latent.shape
            original_latent_shapes.append((t_actual, h_actual, w_actual))

            # Compute padded dimensions (must be divisible by p)
            h_padded = ((h_actual + p - 1) // p) * p
            w_padded = ((w_actual + p - 1) // p) * p

            # Zero-pad if dimensions are not divisible by p
            if h_padded != h_actual or w_padded != w_actual:
                padded = torch.zeros(
                    (self.latent_channel, t_actual, h_padded, w_padded),
                    device=latent.device,
                    dtype=latent.dtype,
                )  # [C,T,H_padded,W_padded]
                padded[:, :, :h_actual, :w_actual] = latent
                latent = padded  # [C,T,H_padded,W_padded]

            # Compute number of patches after padding
            h_patches = h_padded // p
            w_patches = w_padded // p

            # Patchify
            latent = latent.reshape(
                self.latent_channel, t_actual, h_patches, p, w_patches, p
            )  # [C,T,h_patches,p,w_patches,p]
            latent = torch.einsum("cthpwq->thwpqc", latent).reshape(
                -1, p * p * self.latent_channel
            )  # [T*h_patches*w_patches,patch_latent_dim]
            packed_latent.append(latent)

        # We assumed latents we get to the network is already noised
        packed_latent = torch.cat(packed_latent, dim=0)  # [total_vision_patches,patch_latent_dim]
        return packed_latent, original_latent_shapes

    def unpatchify_and_unpack_latents(
        self,
        packed_mse_preds: torch.Tensor,
        token_shapes_vision: List[Tuple[int, int, int]],
        noisy_frame_indexes_vision: list[torch.Tensor],
        original_latent_shapes: List[Tuple[int, int, int]] | None = None,
    ) -> list[torch.Tensor]:
        p = self.latent_patch_size
        unpatchified_latents = []

        # Split packed_mse_preds back into individual latents based on token_shapes_vision
        start_idx = 0
        for i, (t_c, h_c, w_c) in enumerate(token_shapes_vision):
            # Get original shape for unpadding (if provided)
            if original_latent_shapes is not None:
                _, h_orig, w_orig = original_latent_shapes[i]
                # Compute padded dimensions used during patchify
                h_padded = ((h_orig + p - 1) // p) * p
                w_padded = ((w_orig + p - 1) // p) * p
                h_patches = h_padded // p
                w_patches = w_padded // p
            else:
                # Fallback: use token shapes directly (assumes no padding was needed)
                h_orig, w_orig = h_c * p, w_c * p
                h_patches, w_patches = h_c, w_c

            # noisy_frame_indexes_vision is a list of tensors, each with shape (T,),
            # where the values are the noisy frame indices.
            noisy_frame_indexes = noisy_frame_indexes_vision[i]
            t_n = len(noisy_frame_indexes)

            # Initialize with the original shape (after unpadding), zeros for clean frames
            output_tensor = torch.zeros(
                (self.latent_channel, t_c, h_orig, w_orig),
                device=packed_mse_preds.device,
                dtype=packed_mse_preds.dtype,
            )  # [C,T,H_orig,W_orig]
            num_patches = t_n * h_patches * w_patches
            if num_patches > 0:
                end_idx = start_idx + num_patches
                # Extract patches for this latent
                latent_patches = packed_mse_preds[start_idx:end_idx]  # [num_patches,patch_latent_dim]
                # Reshape back to [t_n, h_patches, w_patches, p, p, channels]
                latent_patches = latent_patches.reshape(
                    t_n, h_patches, w_patches, p, p, self.latent_channel
                )  # [T_n,h_patches,w_patches,p,p,C]
                # Invert the einsum operation: "thwpqc->cthpwq"
                latent = torch.einsum("thwpqc->cthpwq", latent_patches)  # [C,T_n,h_patches,p,w_patches,p]
                # Reshape back to [channels, t_n, h_padded, w_padded]
                latent = latent.reshape(
                    self.latent_channel, t_n, h_patches * p, w_patches * p
                )  # [C,T_n,H_padded,W_padded]

                # Crop to original dimensions (unpad the zeros)
                latent = latent[:, :, :h_orig, :w_orig]  # [C,T_n,H_orig,W_orig]

                # Fill only the noisy frame positions using the actual mask indices
                output_tensor[:, noisy_frame_indexes] = latent

                start_idx = end_idx

            unpatchified_latents.append(output_tensor.unsqueeze(0))  # [1,C,T,H,W]

        # Return list of unpatchified latents (supports variable shapes)
        return unpatchified_latents

    def pack_objectflow_latents(
        self,
        tokens_objectflow: list[torch.Tensor],
        token_shapes_objectflow: list[tuple[int, int, int]],
    ) -> torch.Tensor:
        """Pack object-flow latents into a 1D sequence for transformer input.

        Object-flow latents are ``[F, N, 3]`` and are packed frame-major so that
        token index = frame * N + point (matching Stage 3's packing order). Because
        ``[F, N, 3]`` is already stored frame-outer / point-inner, this is a flat
        reshape to ``[F * N, 3]`` per sample.

        Args:
            tokens_objectflow: List of object-flow tensors, each ``[F, N, 3]``.
            token_shapes_objectflow: List of ``(F, N, 1)`` tuples per sample.

        Returns:
            Packed tensor of shape ``[total_objectflow_tokens, 3]``.
        """
        packed = []
        for flow, shape in zip(tokens_objectflow, token_shapes_objectflow):
            num_frames, num_points = shape[0], shape[1]
            # flow: [F, N, 3] -> frame-major flatten [F*N, 3] (frame outer, point inner).
            flow_tokens = flow[:num_frames, :num_points].reshape(num_frames * num_points, flow.shape[-1])
            packed.append(flow_tokens)  # [F*N,3]
        return torch.cat(packed, dim=0)  # [total_objectflow_tokens,3]

    def unpack_objectflow_latents(
        self,
        packed_objectflow_preds: torch.Tensor,
        token_shapes_objectflow: list[tuple[int, int, int]],
        noisy_frame_indexes_objectflow: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        """Unpack object-flow predictions back into per-sample ``[F, N, 3]`` tensors.

        Conditioning is per-FRAME: every noisy frame contributes all N of its point
        tokens, so a sample with ``T_n`` noisy frames supplies ``T_n * N`` packed
        prediction rows (frame-major). Predictions are placed at the noisy frame
        positions; clean frames are left as zeros.

        Args:
            packed_objectflow_preds: Packed predictions ``[total_noisy_tokens, 3]``.
            token_shapes_objectflow: List of ``(F, N, 1)`` tuples per sample.
            noisy_frame_indexes_objectflow: List of tensors, each ``(T_n,)`` holding the
                noisy flow-frame indices for that sample.

        Returns:
            List of per-sample tensors, each ``[F, N, 3]``, with predictions placed at
            noisy frames and zeros at clean frames.
        """
        unpacked = []
        start_idx = 0
        for shape, noisy_frame_indexes in zip(token_shapes_objectflow, noisy_frame_indexes_objectflow):
            num_frames, num_points = shape[0], shape[1]
            output = torch.zeros(
                (num_frames, num_points, self.objectflow_dim),
                device=packed_objectflow_preds.device,
                dtype=packed_objectflow_preds.dtype,
            )  # [F,N,3]

            t_n = len(noisy_frame_indexes)
            if t_n > 0:
                end_idx = start_idx + t_n * num_points
                # packed rows are frame-major: [T_n * N, 3] -> [T_n, N, 3] at noisy frames.
                output[noisy_frame_indexes] = packed_objectflow_preds[start_idx:end_idx].reshape(
                    t_n, num_points, self.objectflow_dim
                )
                start_idx = end_idx

            unpacked.append(output)
        return unpacked

    def _encode_text(
        self,
        packed_seq: PackedSequence,
    ) -> tuple[torch.Tensor, torch.dtype]:
        """Embed text tokens and initialize packed_sequence.

        Args:
            packed_seq: PackedSequence containing text_ids and text_indexes.

        Returns:
            tuple of (packed_sequence, target_dtype) where packed_sequence has text embeddings filled in.
        """
        packed_text_embedding = self.language_model.model.embed_tokens(packed_seq.text_ids)  # [N_text,hidden_size]
        packed_sequence = packed_text_embedding.new_zeros(
            size=(packed_seq.sequence_length, self.hidden_size)
        )  # [N_total,hidden_size]
        packed_sequence[packed_seq.text_indexes] = (
            packed_text_embedding  # [N_text,hidden_size] scattered into [N_total,hidden_size]
        )
        return packed_sequence, packed_text_embedding.dtype

    def _encode_vision(
        self,
        packed_seq: PackedSequence,
        packed_sequence: torch.Tensor,
        target_dtype: torch.dtype,
    ) -> List[Tuple[int, int, int]] | None:
        """Project vision tokens and fill into packed_sequence.

        Args:
            packed_seq: PackedSequence containing vision tokens and metadata.
            packed_sequence: The packed sequence tensor to fill vision embeddings into (modified in-place).
            target_dtype: Target dtype for embeddings (typically from text embedding).

        Returns:
            Original latent shapes before padding (for unpadding during decode), or None if no vision tokens.
        """
        if packed_seq.vision is None or packed_seq.vision.tokens is None:
            # No vision tokens in this batch
            return None

        vision = packed_seq.vision
        assert vision.tokens is not None  # Type narrowing (checked above but reassignment loses it)
        assert vision.token_shapes is not None
        assert isinstance(vision.sequence_indexes, torch.Tensor)
        assert isinstance(vision.timesteps, torch.Tensor)
        torch._assert(
            vision.timesteps.dtype in (torch.long, torch.float32),
            f"Timestep must be long/float32, got {vision.timesteps.dtype}",
        )

        assert isinstance(vision.mse_loss_indexes, torch.Tensor)

        packed_tokens_vision, original_latent_shapes = self.patchify_and_pack_latents(
            vision.tokens, vision.token_shapes
        )  # packed_tokens_vision: [total_vision_patches,patch_latent_dim]
        packed_tokens_vision = self.vae2llm(packed_tokens_vision.to(target_dtype))  # [total_vision_patches,hidden_size]

        has_noisy_vision = vision.mse_loss_indexes.numel() > 0

        if has_noisy_vision:
            timesteps_vision = vision.timesteps.to(dtype=torch.float32) * self.timestep_scale  # [N_noisy_frames_vision]

            # Timesteps are computed in FP32 for numerical stability.
            with torch.autocast("cuda", enabled=True, dtype=torch.float32):
                packed_timestep_embeds_vision = self.time_embedder(
                    timesteps_vision
                )  # [N_noisy_frames_vision,hidden_size]
            packed_timestep_embeds_vision = packed_timestep_embeds_vision.to(
                target_dtype
            )  # [N_noisy_frames_vision,hidden_size]

            packed_tokens_vision = _apply_timestep_embeds_to_noisy_tokens(
                packed_tokens=packed_tokens_vision,
                packed_timestep_embeds=packed_timestep_embeds_vision,
                noisy_frame_indexes=vision.noisy_frame_indexes,
                token_shapes=vision.token_shapes,
            )  # [total_vision_patches,hidden_size]

        packed_sequence[vision.sequence_indexes] = (
            packed_tokens_vision  # [total_vision_patches,hidden_size] scattered into [N_total,hidden_size]
        )
        return original_latent_shapes

    def _encode_objectflow(
        self,
        packed_seq: PackedSequence,
        packed_sequence: torch.Tensor,
        target_dtype: torch.dtype,
    ) -> None:
        """Encode object-flow tokens and fill into packed_sequence.

        Mirrors ``_encode_sound`` (plain-Linear projection + learned modality embed +
        timestep embeds on noisy tokens), with ONE addition: a per-point geometry PE
        derived from each sample's frame-0 absolute xyz is broadcast across all F frames
        and added to the projected tokens, preserving frame-major token order.

        Args:
            packed_seq: PackedSequence containing object-flow tokens and metadata.
            packed_sequence: The packed sequence tensor to fill object-flow embeddings into
                (modified in-place).
            target_dtype: Target dtype for embeddings (typically from text embedding).
        """
        if packed_seq.objectflow is None or packed_seq.objectflow.tokens is None:
            # No object-flow tokens in this batch
            return

        objectflow = packed_seq.objectflow
        assert objectflow.token_shapes is not None
        assert isinstance(objectflow.sequence_indexes, torch.Tensor)
        assert isinstance(objectflow.timesteps, torch.Tensor)
        assert isinstance(objectflow.mse_loss_indexes, torch.Tensor)

        # Pack object-flow latents: list of [F, N, 3] -> [total_tokens, 3] (frame-major).
        packed_tokens_objectflow = self.pack_objectflow_latents(
            objectflow.tokens, objectflow.token_shapes
        )  # [total_objectflow_tokens,objectflow_dim]
        packed_tokens_objectflow = packed_tokens_objectflow.to(target_dtype)

        # Project object-flow tokens + modality embedding. Position info comes from
        # mRoPE position IDs in the attention layers.
        packed_tokens_objectflow = (
            self.objectflow2llm(packed_tokens_objectflow) + self.objectflow_modality_embed
        )  # [total_objectflow_tokens,hidden_size]

        # Per-point geometry PE: constant across frames, added to every frame's point token.
        # Computed per sample from frame-0 absolute xyz, then broadcast frame-major to match
        # the packed token order (token = frame * N + point).
        point_src_list = packed_seq.objectflow_points_src
        assert point_src_list is not None, "objectflow_points_src must be threaded to the network for the per-point PE."
        assert len(point_src_list) == len(objectflow.token_shapes), (
            "objectflow_points_src count must match the number of object-flow samples."
        )
        point_pe_blocks = []
        for points_src, shape in zip(point_src_list, objectflow.token_shapes):
            num_frames, num_points = shape[0], shape[1]
            point_pe = self.objectflow_point_pe(points_src.to(target_dtype))  # [N,hidden_size]
            # Broadcast across F frames -> [F, N, hidden] -> frame-major [F*N, hidden].
            point_pe = point_pe.unsqueeze(0).expand(num_frames, num_points, self.hidden_size)
            point_pe_blocks.append(point_pe.reshape(num_frames * num_points, self.hidden_size))
        packed_point_pe = torch.cat(point_pe_blocks, dim=0)  # [total_objectflow_tokens,hidden_size]
        packed_tokens_objectflow = packed_tokens_objectflow + packed_point_pe

        has_noisy_objectflow = objectflow.mse_loss_indexes.numel() > 0
        if has_noisy_objectflow:
            timesteps_objectflow = objectflow.timesteps * self.timestep_scale  # [N_noisy_flow_frames]
            with torch.autocast("cuda", enabled=True, dtype=torch.float32):
                packed_timestep_embeds_objectflow = self.time_embedder(timesteps_objectflow)
            packed_timestep_embeds_objectflow = packed_timestep_embeds_objectflow.to(target_dtype)

            packed_tokens_objectflow = _apply_timestep_embeds_to_noisy_tokens(
                packed_tokens=packed_tokens_objectflow,
                packed_timestep_embeds=packed_timestep_embeds_objectflow,
                noisy_frame_indexes=objectflow.noisy_frame_indexes,
                token_shapes=objectflow.token_shapes,
            )  # [total_objectflow_tokens,hidden_size]

        packed_sequence[objectflow.sequence_indexes] = packed_tokens_objectflow  # scattered into [N_total,hidden_size]


def _apply_timestep_embeds_to_noisy_tokens(
    packed_tokens: torch.Tensor,
    packed_timestep_embeds: torch.Tensor,
    noisy_frame_indexes: List[torch.Tensor],
    token_shapes: list[tuple[int, ...]],
) -> torch.Tensor:
    """Apply timestep embeddings to noisy tokens.
    Tn is the number of noisy frames for a given sample.
    Tc is the number of clean frames for a given sample.
    T is the total number of frames for a given sample.
    T = Tn + Tc

    Args:
        packed_tokens: The packed tokens to apply timestep embeddings to.
        packed_timestep_embeds: The packed timestep embeddings to apply.
        noisy_frame_indexes: The frame indices to apply timestep embeddings to
            (list of tensors, each with shape (Tn,)).
        token_shapes: The token shapes for each sample. Each entry is a tuple
            shaped like ``(T, ...)`` where trailing dimensions represent the spatial grid.

    Returns:
        The packed tokens with timestep embeddings applied to the noisy tokens.
    """

    # Handle variable token shapes by processing each sample's noisy_frame_indexes individually.
    # The noisy indices are first expanded to cover the entire spatial grid of each frame.
    #
    # For video frames, the spatial grid is (H, W).
    # For action frames, the spatial grid is ().
    # For sound frames, the spatial grid is (1, 1).
    #
    # The noisy indices are then flattened into a single tensor overall. When flattening,
    # we must ensure that the noisy indices from each sample are unique by adding the
    # cumulative sum of the token shapes of previous samples to the noisy indices for
    # a given sample.
    start_noisy_index = 0
    flattened_noisy_frame_indexes = []

    for noisy_indexes_i, token_shape_i in zip(noisy_frame_indexes, token_shapes):
        assert noisy_indexes_i.numel() <= token_shape_i[0]
        spatial_numel_i = math.prod(token_shape_i[1:])
        spatial_indexes_i = torch.arange(spatial_numel_i, device=packed_tokens.device)  # [spatial_numel_i]
        noisy_indexes_i = (
            (noisy_indexes_i * spatial_numel_i).unsqueeze(-1).expand(-1, spatial_numel_i)
        )  # [Tn_i,spatial_numel_i]
        noisy_indexes_i = noisy_indexes_i.clone() + spatial_indexes_i + start_noisy_index  # [Tn_i,spatial_numel_i]
        flattened_noisy_frame_indexes.append(noisy_indexes_i.flatten())  # [Tn_i*spatial_numel_i]
        start_noisy_index += math.prod(token_shape_i)

    flattened_noisy_frame_indexes = torch.cat(flattened_noisy_frame_indexes, dim=0)  # [total_noisy_patches]

    assert packed_tokens.dim() == 2
    assert packed_timestep_embeds.dim() == 2
    assert packed_timestep_embeds.shape[1] == packed_tokens.shape[1]
    assert packed_timestep_embeds.shape[0] <= packed_tokens.shape[0]
    assert flattened_noisy_frame_indexes.dim() == 1
    assert flattened_noisy_frame_indexes.shape[0] == packed_timestep_embeds.shape[0]

    flattened_noisy_frame_indexes = flattened_noisy_frame_indexes.unsqueeze(-1).expand(
        -1,
        packed_tokens.shape[1],
    )  # [total_noisy_patches,hidden_size]

    return packed_tokens.scatter_add(
        dim=0,
        index=flattened_noisy_frame_indexes,
        src=packed_timestep_embeds,
    )  # [total_tokens,hidden_size]
