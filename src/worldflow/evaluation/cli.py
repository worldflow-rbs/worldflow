"""Score a local WorldFlow checkpoint on an indexed held-out JSONL manifest."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from ..data import TrainingDataset
from ..inference.loader import checkpoint_stage, load_model
from ..inference.pipeline import predict_with_model
from .metrics import MetricMeans, PWT_THRESHOLDS, trajectory_metrics


def evaluate(
    checkpoint, manifest, output, *, stage=None, steps=40, seed=42, guidance=None, shift=5.0, limit=None, device=None
):
    checkpoint, manifest, output = Path(checkpoint), Path(manifest), Path(output)
    if output.exists():
        raise FileExistsError(f"Choose a new output directory: {output}")
    if not 0 <= seed < 2**32 or steps <= 0 or (limit is not None and limit < 1):
        raise ValueError("Expected a uint32 seed, a positive --steps and a positive --limit")
    if not math.isfinite(shift) or shift <= 0 or (guidance is not None and not math.isfinite(guidance)):
        raise ValueError("Expected finite guidance and a finite positive shift")
    if device is not None:
        target = torch.device(device)
        if target.type != "cuda":
            raise ValueError("WorldFlow evaluation requires a CUDA device")
        if target.index is not None:
            torch.cuda.set_device(target)
    dataset = TrainingDataset(manifest)
    count = len(dataset) if limit is None else min(limit, len(dataset))
    stage = checkpoint_stage(checkpoint, stage)
    guidance = guidance if guidance is not None else (1.0 if stage == 1 else 2.0)
    model = load_model(checkpoint, stage=stage)

    overall = MetricMeans()
    by_source = defaultdict(MetricMeans)
    ordinals = Counter()
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix=".worldflow-evaluation-") as tmp:
        report = Path(tmp) / "report"
        report.mkdir()
        with (report / "samples.jsonl").open("w", encoding="utf-8") as rows:
            for index in range(count):
                sample = dataset[index]
                source = sample["source"]
                source_index = sample.get("source_index", ordinals[source])
                ordinals[source] += 1
                if type(source_index) is not int or source_index < 0 or seed + source_index >= 2**32:
                    raise ValueError(f"Invalid source index or derived seed for {sample['sample_id']}")
                rgb = np.asarray(sample["video"][0])
                flat_pixels = np.asarray(sample["pixel_indices"])
                uv = np.column_stack((flat_pixels % rgb.shape[1], flat_pixels // rgb.shape[1]))
                result = predict_with_model(
                    model,
                    rgb,
                    np.asarray(sample["points_xyz"]),
                    uv,
                    sample["instruction"],
                    source,
                    stage=stage,
                    seed=seed + source_index,
                    steps=steps,
                    guidance=guidance,
                    shift=shift,
                    prompt_context=sample.get("prompt_context"),
                )
                target = np.asarray(sample["flow_m"])[:, result["selected_indices"], :]
                metrics = trajectory_metrics(result["displacements"], target)
                record = {
                    "index": index,
                    "source": source,
                    "source_index": source_index,
                    "sample_id": sample["sample_id"],
                    "seed": seed + source_index,
                    **metrics,
                }
                rows.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                overall.add(metrics)
                by_source[source].add(metrics)
                print(f"[{index + 1}/{count}] {source}/{sample['sample_id']}  ADE {metrics['ade']:.5f} m", flush=True)
        summary = {
            "manifest": manifest.name,
            "checkpoint": checkpoint.name,
            "stage": stage,
            "seed": seed,
            "steps": steps,
            "guidance": guidance,
            "shift": shift,
            "thresholds_m": list(PWT_THRESHOLDS.values()),
            "aggregation": "equal weight per clip",
            "overall": overall.result(),
            "by_source": {s: by_source[s].result() for s in sorted(by_source)},
        }
        (report / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        report.rename(output)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Local WorldFlow weights directory")
    parser.add_argument("--stage", type=int, choices=(1, 2), help="Generator stage (default in a weights directory: 2)")
    parser.add_argument("--manifest", type=Path, required=True, help="Held-out JSONL in the WorldFlow data format")
    parser.add_argument("--output", type=Path, required=True, help="New evaluation output directory")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42, help="Base noise seed (per clip: base + within-source index)")
    parser.add_argument("--guidance", type=float, help="Text guidance scale (default: 1 for stage 1, 2 for stage 2)")
    parser.add_argument("--shift", type=float, default=5.0)
    parser.add_argument("--limit", type=int, help="Evaluate the first N manifest rows")
    parser.add_argument("--device", help="CUDA device, e.g. cuda:1 (default: current CUDA device)")
    args = parser.parse_args()
    try:
        summary = evaluate(**vars(args))
    except (FileNotFoundError, FileExistsError, ValueError) as error:
        parser.exit(2, f"worldflow-evaluate: {error}\n")
    metrics = summary["overall"]
    print(
        f"{metrics['samples']} clips: ADE {metrics['ade']:.5f} m, FDE {metrics['fde']:.5f} m, PWT {metrics['pwt']:.4f}"
    )


if __name__ == "__main__":
    main()
