# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""MolmoMotion-1M object tracks and per-interval frame-zero camera transformations."""

import pickle
import zipfile
from pathlib import Path

import numpy as np

from .common import FRAMES, canonical_geometry, decode_rgb, training_rgb


class _ArrayUnpickler(pickle.Unpickler):
    # Original archives store dictionaries of numeric arrays in object-typed NPY members.
    # Restrict reconstruction to NumPy primitives, without executing arbitrary pickle globals.
    def find_class(self, module, name):
        if module in ("numpy.core.multiarray", "numpy._core.multiarray") and name in ("_reconstruct", "scalar"):
            return getattr(np.core.multiarray, name)
        if module == "numpy" and name in ("ndarray", "dtype"):
            return getattr(np, name)
        raise pickle.UnpicklingError(f"unsupported object in Molmo track archive: {module}.{name}")


def _load_npz(path):
    result = {}
    with zipfile.ZipFile(path) as archive:
        for member in archive.namelist():
            if not member.endswith(".npy"):
                continue
            with archive.open(member) as stream:
                version = np.lib.format.read_magic(stream)
                reader = (
                    np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
                )
                _, _, dtype = reader(stream)
                if dtype.hasobject:
                    value = _ArrayUnpickler(stream).load()
                else:
                    stream.seek(0)
                    value = np.load(stream, allow_pickle=False)
                result[member[:-4]] = value
    return result


def _uniform_indices(start: int, end: int, count: int = FRAMES) -> np.ndarray:
    return np.rint(np.linspace(start, end, count)).astype(np.int64)


def interval_to_49(start: int, end: int, *, t_min: int = 8, stride: int = 1) -> tuple[np.ndarray, bool, int]:
    """Return source indices, tail-padding flag and true interval length."""

    if stride <= 0 or end < start:
        raise ValueError(f"invalid interval [{start}, {end}] or stride={stride}")
    start, end = start // stride, end // stride
    length = end - start + 1
    if length < t_min:
        raise ValueError("interval_too_short")
    if length > FRAMES:
        return _uniform_indices(start, end), False, length
    indices = np.arange(start, end + 1, dtype=np.int64)
    return np.pad(indices, (0, FRAMES - length), mode="edge"), length < FRAMES, length


def _load_object_tracks(
    dataset: str,
    root: Path,
    file_id: str,
    object_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int]]:
    """Return points ``[N,T,3]``, visibility ``[N,T]``, 2D pixels ``[N,T,2]`` and ``(H,W)``."""

    if dataset == "droid":
        base = root / dataset / "tracks"
        d3, d2 = _load_npz(base / f"{file_id}_3d.npz"), _load_npz(base / f"{file_id}_2d.npz")
        points = np.asarray(d3["points_3d"], dtype=np.float32).transpose(1, 0, 2)
        visibility = np.asarray(d3["valid_3d"], dtype=bool).T
        pixels = np.asarray(d2["tracks_2d"], dtype=np.float32).transpose(1, 0, 2)
        dim = tuple(int(x) for x in np.asarray(d2["ds_dim"]).tolist())
        return points, visibility, pixels, dim
    if dataset == "molmospaces":
        base = root / dataset / "tracks"
        d3, d2 = _load_npz(base / f"{file_id}_3d.npz"), _load_npz(base / f"{file_id}_2d.npz")
        points_dict, vis_dict = d3["points_3d"].item(), d3["visibility"].item()
        pixels_dict = d2["tracks"].item()
        if object_name not in points_dict or object_name not in pixels_dict:
            raise KeyError(f"object {object_name!r} missing from MolmoSpaces tracks")
        points = np.asarray(points_dict[object_name], dtype=np.float32)[:, ::4, :]
        visibility = np.asarray(vis_dict[object_name], dtype=bool)[:, ::4]
        pixels = np.asarray(pixels_dict[object_name], dtype=np.float32)[::4].transpose(1, 0, 2)
        dim = tuple(int(x) for x in np.asarray(d2["dim"]).tolist())
        return points, visibility, pixels, dim
    if dataset in {"egodex", "ytvis", "stereo4d"}:
        base = root / dataset / "tracks"
        if dataset == "egodex":
            base = base / "object"
        d3, d2 = _load_npz(base / f"{file_id}_3d.npz"), _load_npz(base / f"{file_id}_2d.npz")
        points_raw, vis_raw = d3["points_3d"], d3.get("visibility")
        pixels_raw = d2["tracks"]
        points_dict = points_raw.item() if points_raw.dtype == object else None
        vis_dict = vis_raw.item() if vis_raw is not None and vis_raw.dtype == object else None
        pixels_dict = pixels_raw.item() if pixels_raw.dtype == object else None
        if points_dict is not None:
            if object_name not in points_dict or pixels_dict is None or object_name not in pixels_dict:
                raise KeyError(f"object {object_name!r} missing from {dataset} tracks")
            points = np.asarray(points_dict[object_name], dtype=np.float32)
            visibility = (
                np.asarray(vis_dict[object_name], dtype=bool)
                if vis_dict is not None and object_name in vis_dict
                else np.isfinite(points).all(axis=-1)
            )
            pixels = np.asarray(pixels_dict[object_name], dtype=np.float32)
            # Released 2D dicts are (T,N,2); normalize to (N,T,2).
            if pixels.ndim == 3 and pixels.shape[0] == points.shape[1]:
                pixels = pixels.transpose(1, 0, 2)
        else:
            points = np.asarray(points_raw, dtype=np.float32)
            visibility = np.asarray(vis_raw, dtype=bool) if vis_raw is not None else np.isfinite(points).all(axis=-1)
            if visibility.ndim == 3 and visibility.shape[-1] == 1:
                visibility = visibility[..., 0]
            pixels = np.asarray(pixels_raw, dtype=np.float32)
            if pixels.ndim == 3 and pixels.shape[0] == points.shape[1]:
                pixels = pixels.transpose(1, 0, 2)
        dim = tuple(int(x) for x in np.asarray(d2["dim"]).tolist())
        return points, visibility, pixels, dim
    if dataset == "xperience":
        path = root / dataset / "tracks" / f"{file_id}_object.npz"
        data = _load_npz(path)
        points = np.asarray(data["points_3d"], dtype=np.float32)
        visibility = np.asarray(data["visibility"], dtype=bool)
        pixels_raw = data.get("tracks_2d", data.get("pixel_coords"))
        if pixels_raw is None:
            raise KeyError(f"{path} has no tracks_2d/pixel_coords")
        pixels = np.asarray(pixels_raw, dtype=np.float32)
        # Xperience stores object tracks as (N,T,2), already canonical order.
        return points, visibility, pixels, (512, 512)
    raise ValueError(f"unknown MolmoMotion dataset {dataset!r}")


def _pose_at_index(data: np.ndarray, inds: np.ndarray | None, index: int) -> np.ndarray:
    """Select a pose row using annotation-frame ids when available."""

    if inds is not None:
        matches = np.flatnonzero(np.asarray(inds) == int(index))
        if matches.size:
            return np.asarray(data[int(matches[0])], dtype=np.float32)
    clipped = min(max(int(index), 0), int(len(data) - 1))
    return np.asarray(data[clipped], dtype=np.float32)


def _apply_w2c(points: np.ndarray, w2c: np.ndarray) -> np.ndarray:
    homogeneous = np.concatenate([points, np.ones((*points.shape[:2], 1), dtype=np.float32)], axis=-1)
    return np.einsum("ij,ntj->nti", w2c, homogeneous)[..., :3]


def _frame0_camera(
    dataset: str,
    root: Path,
    file_id: str,
    points: np.ndarray,
    frame_index: int = 0,
    temporal_stride: int = 1,
) -> np.ndarray:
    if dataset == "droid":
        return points
    if dataset == "molmospaces":
        camera = _load_npz(root / dataset / "camera" / f"{file_id}.npz")
        raw_index = int(frame_index) * temporal_stride
        c2w = _pose_at_index(np.asarray(camera["cam_poses"], dtype=np.float32), None, raw_index)
        w2c = np.linalg.inv(c2w)
        return _apply_w2c(points, w2c)
    if dataset in {"egodex", "ytvis"}:
        camera = _load_npz(root / dataset / "camera" / "pose" / f"{file_id}.npz")
        c2w = _pose_at_index(np.asarray(camera["data"], dtype=np.float32), camera.get("inds"), frame_index)
        w2c = np.linalg.inv(c2w)
        return _apply_w2c(points, w2c)
    if dataset == "xperience":
        pose_path = root / dataset / "camera" / "poses" / f"{file_id}.npz"
        # Reconstruction normalizes clip frame 0 to identity.  Without the
        # optional pose artifact this is sufficient only when the interval
        # itself starts at frame 0; later intervals need c2w[t0] to rebase.
        if not pose_path.exists():
            if int(frame_index) == 0:
                return points
            raise ValueError("missing_camera_for_interval_rebase")
        camera = _load_npz(pose_path)
        c2w = _pose_at_index(np.asarray(camera["c2w"], dtype=np.float32), None, frame_index)
        w2c = np.linalg.inv(c2w)
        return _apply_w2c(points, w2c)
    if dataset == "stereo4d":
        camera = _load_npz(root / dataset / "camera" / f"{file_id}.npz")
        pose_index = min(max(int(frame_index), 0), len(camera["cam_pos"]) - 1)
        pos = np.asarray(camera["cam_pos"], dtype=np.float32)[pose_index]
        forward = np.asarray(camera["cam_fwd"], dtype=np.float32)[pose_index]
        z = forward / (np.linalg.norm(forward) + 1e-9)
        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        x = np.cross(up, z)
        if np.linalg.norm(x) < 1e-6:
            raise ValueError("stereo4d_degenerate_camera_frame")
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        w2c = np.eye(4, dtype=np.float32)
        w2c[:3, :3] = np.stack([x, y, z], axis=0)
        w2c[:3, 3] = -w2c[:3, :3] @ pos
        return _apply_w2c(points, w2c)
    raise NotImplementedError(dataset)


def load_molmo(row, root):
    dataset = row["source"].removeprefix("molmo_")
    file_id, obj = row["file"], row["object_name"]
    if "hand" in obj.lower():
        raise ValueError("Molmo hand tracks are excluded from object motion training")
    stride = 4 if dataset == "molmospaces" else 1
    start, end = row["interval"]
    indices, _, _ = interval_to_49(int(start), int(end), stride=stride)
    tracks, visible, pixels, native_hw = _load_object_tracks(dataset, root, file_id, obj)
    tracks = _frame0_camera(dataset, root, file_id, tracks[:, indices], int(indices[0]), stride)
    uv = pixels[:, indices]
    valid = visible[:, indices[0]] & np.isfinite(uv).all(axis=(1, 2))
    identity = f"{dataset}:{file_id}:{obj}:{row['interval']}"
    geometry = canonical_geometry(tracks[valid].transpose(1, 0, 2), uv[valid, 0], native_hw, identity)
    rgb, _ = decode_rgb(root / dataset / "videos" / f"{file_id}.mp4", indices * stride)
    return {**geometry, "video": training_rgb(rgb)}
