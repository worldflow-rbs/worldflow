# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Index native WorldFlow training sources or materialize portable numeric examples."""

import argparse
import hashlib
import json
import os
import sqlite3
import tempfile
import uuid
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from worldflow.inference.presets import SOURCE_DOMAINS
from .catalog import OPTIONAL_KEYS, native_records
from .dataset import read_sample


def published_sources(stage):
    demo = "datasets/demo5"
    run = "metadata/objectflow_run"
    qc = "metadata/qc"
    if stage == 1:
        return {"sources": [{"format": "demo5_raw", "root": demo}]}
    sim = "datasets/simulation"
    contract = f"{qc}/domain_scale_v3_simfix_20260916/objectflow_training_contract.json"
    sources = [
        {
            "format": "demo5",
            "root": demo,
            "manifest": f"{run}/demo5_manifests/snapshots/demo5-en-20260903T140412Z-51c4ad14/training.jsonl",
        }
    ]
    for subset in ("maniskill", "mikasa", "metaworld", "metafold", "rlbench", "robocasa", "robotwin"):
        spec = {"format": "sim7", "subset": subset}
        if subset in ("maniskill", "mikasa", "metaworld"):
            spec["root"] = f"{sim}/webdataset/sim_v1_wbs_v0.15/{subset}/flatten"
        elif subset == "robocasa":
            spec["root"] = f"{sim}/packed_shards_new/robocasa/flat"
            spec["qc"] = spec["root"] + "/qc_skip_flags.jsonl"
        else:
            spec["root"] = (
                f"{sim}/webdataset/sim_v1_wbs_v0.16/robotwin/flatten"
                if subset == "robotwin"
                else f"{sim}/packed_shards/{subset}"
            )
            spec["qc"] = f"{qc}/sim_v1_wbs_v0.16/{subset}_skip_flags.jsonl"
            spec["skip_statuses"] = "all" if subset == "robotwin" else ["fail"]
        sources.append(spec)
    for name in ("agibot", "sthv2", "epic"):
        sources.append({"format": "any4d", "source": name, "root": "datasets/realworld"})
    for name in ("droid", "egodex", "molmospaces", "stereo4d", "xperience", "ytvis"):
        directory = {"molmospaces": "domain_scale_v3_simfix_20260916", "ytvis": "outdoor_qc_v1_20260916"}.get(
            name, "molmo_scale_qc_v1_20260915"
        )
        sources.append(
            {
                "format": "molmo",
                "root": "datasets/molmo_motion_1m",
                "name": "molmo_" + name,
                "manifest": f"{qc}/{directory}/{name}.jsonl",
            }
        )
    return {
        "contract": contract,
        "sources": sources,
        "exclude": [
            f"{run}/objectflow_benchmark_v1.json",
            f"{qc}/real3_qc_demo5_aligned_v1_20260915/real3_qc_exclude.json",
        ],
    }


def verify_file(path, expected):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected:
        raise ValueError(f"source metadata differs from the published hash: {path}")


def exclusions(paths, config_base):
    result = defaultdict(set)
    for path in paths:
        data = json.loads((config_base / path).read_text())
        if "samples" in data:
            for row in data["samples"]:
                result[row["source"]].add(row["sample_id"])
        else:
            for source, ids in data.items():
                if not isinstance(ids, list) or any(not isinstance(s, str) for s in ids):
                    raise ValueError(f"invalid exclusion list in {path}")
                result[source].update(ids)
    return result


def prepare(config, config_base, output, *, limit=None, validate=False, materialize=False):
    if limit is not None and limit < 1:
        raise ValueError("limit-per-source must be positive")
    config_base, output = Path(config_base).resolve(), Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if not config["sources"]:
        raise ValueError("need at least one source specification")
    if config.get("contract"):
        contract = json.loads((config_base / config["contract"]).read_text())
        benchmark, qc = config["exclude"]
        verify_file(config_base / benchmark, contract["benchmark_sha256"])
        verify_file(config_base / qc, contract["real3_qc_sha256"])
        hashes = {
            row["subset"]: row["qc_manifest_sha256"] for row in contract["subsets"] if row.get("qc_manifest_sha256")
        }
        for spec in config["sources"]:
            if spec["format"] == "molmo":
                verify_file(config_base / spec["manifest"], hashes[spec["name"]])
            if spec["format"] == "sim7" and "qc" in spec:
                expected = {
                    "metafold": "01c08b195778d23c93585e4bfe028d1e3f6dde7bf7c8d3572f0d2d467eb9803f",
                    "rlbench": "ec1a10d0a5a385c9bb37fbcb872aabffe56da461993cd574bfa2839347ac6086",
                    "robocasa": "c848759c22dcb38225c911521739871bae1746d52839acbcf252817339dde57a",
                    "robotwin": "466accf89f1125436a609c9d5258c3d67699b3a741455acde7b260d97cc4b5f2",
                }
                verify_file(config_base / spec["qc"], expected[spec["subset"]])
    excluded = exclusions(config.get("exclude", []), config_base)
    counts = Counter()
    with tempfile.TemporaryDirectory(dir=output.parent) as workspace:
        temp = Path(workspace)
        record_dir = temp / (output.stem + "-samples-" + uuid.uuid4().hex[:12])
        if materialize:
            record_dir.mkdir()
        # Million-clip catalogs can share IDs across datasets. Keep the uniqueness index out of host RAM.
        db = sqlite3.connect(temp / "seen.sqlite")
        try:
            db.execute("CREATE TABLE records (src TEXT, id TEXT, PRIMARY KEY(src,id)) WITHOUT ROWID")
            with (temp / "manifest.jsonl").open("w", encoding="utf-8") as stream:
                for spec in config["sources"]:
                    accepted = 0
                    for row in native_records(spec, config_base, output.parent):
                        src, identity = row["source"], row["sample_id"]
                        if src not in SOURCE_DOMAINS or not isinstance(identity, str) or not identity:
                            raise ValueError(f"invalid training source or id: {src!r}, {identity!r}")
                        if identity in excluded[src]:
                            continue
                        try:
                            db.execute("INSERT INTO records VALUES(?,?)", (src, identity))
                        except sqlite3.IntegrityError as exc:
                            raise ValueError(f"duplicate {src} record {identity!r}") from exc
                        if validate or materialize:
                            sample = read_sample(row, output.parent)
                        if materialize:
                            key = hashlib.sha256(json.dumps([src, identity]).encode()).hexdigest()
                            path = record_dir / key[:2] / (key + ".npz")
                            path.parent.mkdir(exist_ok=True)
                            np.savez_compressed(
                                path,
                                **{k: sample[k].numpy() for k in ("video", "points_xyz", "pixel_indices", "flow_m")},
                            )
                            row = {
                                "format": "prepared",
                                "sample": str(path.relative_to(temp)),
                                "source": src,
                                "sample_id": identity,
                                "instruction": sample["instruction"],
                                **{k: sample[k] for k in OPTIONAL_KEYS if k in sample},
                            }
                        stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                        counts[src] += 1
                        accepted += 1
                        if limit is not None and accepted >= limit:
                            break
        finally:
            db.close()
        if not counts:
            raise ValueError("no eligible training records")
        # Each completed manifest owns a distinct, immutable sample directory.
        if materialize:
            record_dir.rename(output.parent / record_dir.name)
        os.replace(temp / "manifest.jsonl", output)
    return dict(counts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--root", type=Path, help="root containing the published datasets/ and metadata/ trees")
    mode.add_argument("--config", type=Path, help="JSON source specification for custom layouts and corpora")
    parser.add_argument("--stage", choices=(1, 2), type=int, help="published training stage for --root (default: 2)")
    parser.add_argument("--output", required=True, type=Path, help="write the JSONL training manifest here")
    parser.add_argument("--limit-per-source", type=int, help="maximum eligible clips per input source specification")
    parser.add_argument("--validate", action="store_true", help="decode and check each admitted clip")
    parser.add_argument(
        "--materialize", action="store_true", help="write portable numeric NPZ files alongside the manifest"
    )
    args = parser.parse_args()
    if args.config:
        if args.stage:
            parser.error("--stage selects a published --root recipe")
        config = json.loads(args.config.read_text())
        base = args.config.resolve().parent
    else:
        config, base = published_sources(args.stage or 2), args.root
    counts = prepare(
        config, base, args.output, limit=args.limit_per_source, validate=args.validate, materialize=args.materialize
    )
    print(json.dumps({"manifest": str(args.output), "counts": counts}, indent=2))


if __name__ == "__main__":
    main()
