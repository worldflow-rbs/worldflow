# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Decode a SAM2 object from an Any4D 10-bit reference-delta flow video."""

import json
import subprocess

import numpy as np

from .common import canonical_geometry, decode_rgb, evenly_spaced, training_rgb
from .points import select_object_indices


def _packed_frames(path, full_width, x0, y0, width, height):
    # Use the native 16-bit RGB conversion used to create the reference flow arrays.
    # Switching conversion libraries can change quantized coordinates and point selection.
    # The published QC geometry was decoded inside each mask's bounding box. Cropping
    # before RGB conversion also controls chroma interpolation at its boundary.
    crop = (
        f"[0]crop={width}:{height}:{x0}:{y0}[rgb];[0]crop={width}:{height}:{full_width + x0}:{y0}[aux];[rgb][aux]hstack"
    )
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-nostdin",
        "-i",
        str(path),
        "-filter_complex_threads",
        "1",
        "-filter_complex",
        crop,
        "-threads",
        "1",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb48le",
        "pipe:1",
    ]
    frame_bytes = height * 2 * width * 3 * 2
    with subprocess.Popen(command, stdout=subprocess.PIPE) as process:
        try:
            while buffer := process.stdout.read(frame_bytes):
                if len(buffer) != frame_bytes:
                    raise ValueError("incomplete 16-bit Any4D flow frame")
                yield np.frombuffer(buffer, dtype="<u2").reshape(height, 2 * width, 3)
            if process.wait() != 0:
                raise ValueError(f"FFmpeg failed to decode {path}")
        finally:
            if process.poll() is None:
                process.kill()


def load_any4d(row, base):
    folder = base / row["any4d"]
    semantics = json.loads((folder / "meta.json").read_text())
    expected = {
        "scene_point_video_frame": "world",
        "scene_point_flow_frame": "world",
        "cam_poses_convention": "camera_to_world",
        "reference_frames": [0],
        "num_reference_frames": 1,
    }
    world = semantics["world_frame"]
    if (
        any(semantics.get(k) != v for k, v in expected.items())
        or world["index"] != 0
        or world["convention"] != "camera_frame_at_t0"
    ):
        raise ValueError(f"{row['sample_id']}: unsupported Any4D coordinate convention")
    (metadata,) = folder.glob("scene_point_flow*_h265_crf0.json")
    codec = json.loads(metadata.read_text())
    if codec["encoding"] != "ref_delta_4ch_v3" or codec["bits"] != 10 or not np.isfinite(codec["max_scale"]):
        raise ValueError("unsupported Any4D flow encoding")
    length, height, width = map(int, codec["shape_THW"])
    anchor = np.load(folder / "scene_point_flow_ref00000.anchor.npy", allow_pickle=False)
    with np.load(base / row["mask"], allow_pickle=False) as archive:
        mask = archive[sorted(archive.files)[0]].astype(bool)
    if mask.shape != (height, width) or anchor.shape != (height, width, 3):
        raise ValueError("SAM2 mask and Any4D anchor must match the flow video geometry")
    mask_y, mask_x = np.nonzero(mask)
    if not mask_x.size:
        raise ValueError("SAM2 mask is empty at frame zero")
    x0, y0 = int(mask_x.min()), int(mask_y.min())
    roi_w, roi_h = int(mask_x.max()) - x0 + 1, int(mask_y.max()) - y0 + 1
    ys, xs = np.nonzero(mask & np.isfinite(anchor).all(axis=-1))
    frame_ids = evenly_spaced(length)
    requested = set(frame_ids)
    tracks = {}
    for i, packed in enumerate(_packed_frames(metadata.with_suffix(".mp4"), width, x0, y0, roi_w, roi_h)):
        if i not in requested:
            continue
        median = np.float32(np.median(packed[:, roi_w:, 0] >> 6))
        scale = median / np.float32(1023) * np.float32(codec["max_scale"])
        offsets = ((packed[ys - y0, xs - x0] >> 6).astype(np.float32) / 1023 * 2 - 1) * scale
        xyz = anchor[ys, xs].astype(np.float32) + offsets
        if i == 0:
            selected = select_object_indices(xyz, 64, seed=42, sample_id=row["sample_id"])
            if selected.indices is None:
                raise ValueError(f"{row['sample_id']}: {selected.reason}")
            xs, ys, xyz = xs[selected.indices], ys[selected.indices], xyz[selected.indices]
        tracks[i] = xyz
    if tracks.keys() != requested:
        raise ValueError("truncated Any4D flow video")
    geometry = canonical_geometry(
        np.stack([tracks[i] for i in frame_ids]), np.stack([xs, ys], 1), (height, width), row["sample_id"]
    )
    rgb, _ = decode_rgb(base / row["video"], native_resize=True)
    return {**geometry, "video": training_rgb(rgb)}
