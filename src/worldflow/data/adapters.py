# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Load native training clips from the Sim7, Demo5, Real3 and MolmoMotion data families."""

import io
import json

import numpy as np

from .common import canonical_geometry, decode_rgb, evenly_spaced, training_rgb
from .points import voxel_uniform_idx


def load_native(kind, row, base):
    if kind == "molmo":
        from .molmo import load_molmo

        return load_molmo(row, base / row["root"])
    if kind == "any4d":
        from .realworld import load_any4d

        return load_any4d(row, base)
    if kind in {"demo5", "demo5_raw"}:
        return _demo5(row, base, warm_start=kind == "demo5_raw")
    if kind == "sim7":
        return _sim7(row, base)
    if kind == "tracks":
        with np.load(base / row["tracks"], allow_pickle=False) as data:
            xyz = data["trajectories"]
            uv = data["pixels_uv"]
        frame_ids = evenly_spaced(len(xyz))
        rgb, native_hw = decode_rgb(base / row["video"], frame_ids)
        return {**canonical_geometry(xyz[frame_ids], uv, native_hw, row["sample_id"]), "video": training_rgb(rgb)}
    raise ValueError(f"unknown training record format {kind!r}")


def _demo5(row, base, *, warm_start):
    with np.load(base / row["tracks"], allow_pickle=False) as bundle:
        xyz = bundle["obj_tgt"].astype(np.float32)
        xyz = xyz[evenly_spaced(len(xyz))]
        fallback = bundle["intrinsics"]
    with np.load(base / row["episode"], allow_pickle=False) as raw:
        video = raw["rgb_camera1"]
        rgb = video[evenly_spaced(len(video))]
        k = np.array(raw["intrinsics_camera1"] if "intrinsics_camera1" in raw else fallback, dtype=np.float64)
        instruction = row.get("instruction") or str(raw["prompt"].item())
    h, w = rgb.shape[1:3]
    if k[0, 2] <= 2 and k[1, 2] <= 2:
        k[0] *= w
        k[1] *= h
    if row.get("filter_outliers", True) and xyz.shape[1] > 3 and (warm_start or xyz.shape[1] != 64):
        displacement = xyz.astype(np.float64) - xyz[0:1]
        distance = np.linalg.norm(displacement - np.median(displacement, axis=1, keepdims=True), axis=-1).max(axis=0)
        median = np.median(distance)
        outliers = distance > max(median + 3 * np.median(np.abs(distance - median)), 0.02)
        cap = int(0.3 * len(distance))
        if outliers.sum() > cap:
            outliers[:] = False
            outliers[np.argsort(distance)[::-1][:cap]] = True
        xyz = xyz[:, ~outliers]
    if warm_start:
        if xyz.shape[1] == 0:
            raise ValueError(f"{row['sample_id']}: no object points")
        # The SpatialCog warm-start predates the multisource 8–63 point replacement policy.
        xyz = xyz[:, voxel_uniform_idx(xyz[0], 64)]
    projected = xyz[0].astype(np.float64) @ k.T
    uv = projected[:, :2] / np.clip(projected[:, 2:], 1e-6, None)
    return {
        **canonical_geometry(xyz, uv, (h, w), row["sample_id"]),
        "video": training_rgb(rgb),
        "instruction": instruction,
    }


def _sim7(row, base):
    # The index stores offsets into uncompressed tar members so workers read only this record.
    with (base / row["shard"]).open("rb") as handle:

        def member(name):
            offset, size = row["members"][name]
            if min(offset, size) < 0:
                raise ValueError("invalid tar member bounds")
            handle.seek(offset)
            return io.BytesIO(handle.read(size))

        def array(name):
            return np.load(member("object_mask_points_" + name + ".npy"), allow_pickle=False)

        valid = array("valid")[0].astype(bool)
        xyz = array("tgt")[:, valid]
        initial = array("src")[valid]
        pixels = array("idx")[valid]
        meta = json.load(member("meta.json"))
        rgb, native_hw = decode_rgb(member("rgb.mp4"))
    y, x = np.divmod(pixels, native_hw[1])
    geometry = canonical_geometry(xyz, np.stack((x, y), axis=1), native_hw, row["sample_id"], source_points=initial)
    return {**geometry, "video": training_rgb(rgb), "instruction": meta.get("caption") or meta.get("task_description")}
