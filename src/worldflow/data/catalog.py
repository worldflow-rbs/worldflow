# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Stream native training records and apply source-specific admission rules."""

import gzip
import io
import json
import os
import tarfile
import zipfile
import zlib
from pathlib import Path

import numpy as np


SIM_FIELDS = {
    "rgb.mp4",
    "meta.json",
    *("object_mask_points_" + field + ".npy" for field in ("valid", "src", "tgt", "idx")),
}
SKIP_TASKS = {
    "mikasa": {"ShellGameTouch-v0", "ShellGamePush-v0"},
    "metaworld": {
        "coffee-button-v3",
        "handle-pull-side-v3",
        "handle-pull-v3",
        "peg-unplug-side-v3",
        "reach-wall-v3",
        "stick-pull-v3",
        "stick-push-v3",
    },
}
PATH_KEYS = ("video", "tracks", "episode", "sample", "root", "mask", "any4d", "shard")
OPTIONAL_KEYS = ("prompt_context", "instruction_variants", "source_index")


def json_lines(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def relocate(path, old_base, new_base):
    path = Path(path)
    if path.is_absolute():
        if old_base is None:
            return path
        path = path.relative_to(old_base)
    return new_base / path


def _sim_records(spec, root, config_base):
    subset = spec["subset"]
    skip = set()
    if spec.get("qc"):
        statuses = spec.get("skip_statuses", ["fail"])
        if statuses != "all" and not isinstance(statuses, list):
            raise ValueError("skip_statuses must be 'all' or a list")
        for record in json_lines(config_base / spec["qc"]):
            if statuses == "all" or record["status"] in statuses:
                skip.add((record["shard"], record["sample_key"]))
    shards = sorted(root.glob(spec.get("glob", "shard-*.tar")))
    if not shards:
        raise FileNotFoundError(f"no uncompressed Sim7 shards in {root}")
    for shard in shards:
        with tarfile.open(shard, mode="r:") as tar:
            # Index each shard independently. Native archives occasionally interleave sidecars.
            groups = {}
            for member in tar:
                head, dot, tail = member.name.rpartition("/")
                key, sep, field = tail.partition(".")
                if member.isfile() and sep and field in SIM_FIELDS:
                    full_key = head + dot + key
                    members = groups.setdefault(full_key, {})
                    if field in members:
                        raise ValueError(f"duplicate tar member {member.name}")
                    members[field] = (member.offset_data, member.size)
            for key, members in groups.items():
                if (
                    set(members) != SIM_FIELDS
                    or key.split("/")[0] in SKIP_TASKS.get(subset, set())
                    or (shard.name, key) in skip
                ):
                    continue

                def read(name):
                    offset, size = members[name]
                    tar.fileobj.seek(offset)
                    return tar.fileobj.read(size)

                def array(field):
                    return np.load(io.BytesIO(read(f"object_mask_points_{field}.npy")), allow_pickle=False)

                valid = array("valid")[0].astype(bool)
                if np.count_nonzero(valid) < 8:
                    continue
                start = array("src").astype(np.float32)[valid]
                future = array("tgt").astype(np.float32)[1:, valid]
                peak = np.linalg.norm(future - start, axis=-1).max()
                if not 0.05 <= peak <= 3.0:
                    continue
                meta = json.loads(read("meta.json"))
                instruction = meta.get("caption") or meta.get("task_description")
                if not instruction:
                    continue
                yield {
                    "format": "sim7",
                    "source": "sim7",
                    "sample_id": f"{subset}/{shard.name}:{key}",
                    "members": members,
                    "shard": shard,
                    "instruction": instruction,
                    **{k: meta[k] for k in OPTIONAL_KEYS if k in meta},
                }


def _compatible_mask(path, expected):
    # Scan the NPY headers only. The original complete-segment catalogs include masks
    # that change raster shape partway through a clip or have truncated members.
    try:
        with zipfile.ZipFile(path) as archive:
            names = sorted(archive.namelist())
            if not names:
                return False
            for name in names:
                with archive.open(name) as member:
                    version = np.lib.format.read_magic(member)
                    reader = (
                        np.lib.format.read_array_header_1_0
                        if version == (1, 0)
                        else np.lib.format.read_array_header_2_0
                    )
                    shape, _, dtype = reader(member)
                    if shape != expected or dtype.hasobject:
                        return False
            with archive.open(names[0]) as first:
                return np.count_nonzero(np.load(first, allow_pickle=False)) >= 8
    except (EOFError, OSError, ValueError, zipfile.BadZipFile, zlib.error):
        return False


def _any4d_records(spec, root, manifest):
    source = spec["source"]
    for item in json_lines(manifest or root / source / "_meta/complete_samples.jsonl.gz"):
        modules = ("time_seg", "vlm_anno", "sam2", "any4d")
        if not item.get("seg_name") or not all(item.get("paths", {}).get(module) for module in modules):
            continue
        paths = {}
        for module in modules:
            tail = item["paths"][module].split("/artifacts/")[-1].strip("/")
            paths[module] = root / source / module / tail.removeprefix(module + "/")
        anno = paths["vlm_anno"]
        if anno.suffix.lower() != ".json":
            anno = next(iter(sorted(anno.parent.glob(anno.name + "*_gemini.json"))), None)
        if not anno or not anno.is_file():
            continue
        segments = json.loads(anno.read_text()).get("Level 1", {}).get("Segments", [])
        instruction = segments[0].get("Description", "").strip() if segments else ""
        if not instruction or not paths["time_seg"].is_file():
            continue
        codec = next(paths["any4d"].glob("scene_point_flow*_h265_crf0.json"), None)
        if codec is None:
            continue
        expected = tuple(json.loads(codec.read_text())["shape_THW"][1:])
        for mask_path in sorted(paths["sam2"].rglob("*_masks.npz")):
            if not _compatible_mask(mask_path, expected):
                continue
            identity = item["seg_name"] + "::" + mask_path.stem.removesuffix("_masks")
            yield {
                "format": "any4d",
                "source": source,
                "sample_id": identity,
                "instruction": instruction,
                "video": paths["time_seg"],
                "any4d": paths["any4d"],
                "mask": mask_path,
                **{k: item[k] for k in OPTIONAL_KEYS if k in item},
            }


def _raw_demo5(root):
    bundles = root / "wfm_export"
    preparation = json.loads((bundles / "prep_index.json").read_text())
    if preparation["gt_version"] != 2 or preparation["motion_filter"] != [0.05, 3.0]:
        raise ValueError("expected Demo5 GT v2 and the 5 cm to 3 m motion filter")
    excluded = set()
    for sidecar in (".fixed_val20.json", ".qc_exclude.json"):
        excluded.update(json.loads((bundles / sidecar).read_text())["stems"])
    raw = {}
    for episode in (root / "_t4w_flat").glob("*/episode_*.npz"):
        raw.setdefault(episode.stem, episode)
    if not raw:
        raise FileNotFoundError(f"no native Demo5 episode files under {root}/_t4w_flat")
    for episode in preparation["episodes"]:
        stem = episode["stem"]
        if episode["kept"] and stem not in excluded and stem in raw:
            yield {
                "format": "demo5_raw",
                "source": "demo5",
                "sample_id": stem,
                "episode": raw[stem],
                "tracks": bundles / stem / "sample.npz",
            }


def native_records(spec, config_base, output_base):
    kind = spec["format"]
    root = config_base / spec.get("root", ".")
    manifest = config_base / spec["manifest"] if "manifest" in spec else None
    if kind == "demo5_raw":
        rows = _raw_demo5(root)
    elif kind == "sim7":
        rows = _sim_records(spec, root, config_base)
    elif kind == "any4d":
        rows = _any4d_records(spec, root, manifest)
    else:
        rows = _manifest_records(kind, root, spec, manifest)
    for row in rows:
        for name in PATH_KEYS:
            if name in row:
                row[name] = os.path.relpath(row[name], output_base)
        yield row


def _manifest_records(kind, root, spec, manifest):
    if manifest is None:
        raise ValueError(f"{kind} source requires an input manifest")
    for item in json_lines(manifest):
        if item.get("split", "train") not in {"train", "training"}:
            continue
        optional = {k: item[k] for k in OPTIONAL_KEYS if k in item}
        if kind == "demo5":
            if not item["qc_pass"] or not item["motion_pass"]:
                continue
            # Exported snapshot paths may describe the machine that created the index.
            # Both payloads share a capture root above wfm_export/<episode>/sample.npz.
            bundle = Path(item["bundle_path"])
            original_root = spec.get("original_root")
            if bundle.is_absolute() and original_root is None:
                if bundle.parents[1].name != "wfm_export":
                    raise ValueError("specify original_root to relocate a nonstandard Demo5 export")
                original_root = bundle.parents[2]
            row = {
                "format": kind,
                "source": "demo5",
                "sample_id": item["source_sample_id"],
                "instruction": item["prompt"],
                "episode": relocate(item["raw_video_path"], original_root, root),
                "tracks": relocate(bundle, original_root, root),
            }
        elif kind == "molmo":
            if item.get("reject_reason") or item.get("status", "canonicalized") != "canonicalized":
                continue
            if "hand" in item["object_name"].lower():
                continue
            row = {
                "format": kind,
                "source": "molmo_" + item["dataset"],
                "sample_id": item["source_sample_id"],
                "instruction": item["caption"],
                "root": root,
                **{k: item[k] for k in ("file", "interval", "object_name")},
            }
        elif kind == "manifest":
            row = dict(item)
            for name in PATH_KEYS:
                if name in row:
                    row[name] = manifest.parent / row[name]
        else:
            raise ValueError(f"unknown source specification {kind}")
        yield {**row, **optional}
