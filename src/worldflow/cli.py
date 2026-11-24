"""Predict future RGB video and 3D query-point trajectories from a single image."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from .backend import validate_bundle
from .presets import DOMAIN_SCALES, SOURCE_DOMAINS

DEFAULT_SCENE = Path(__file__).parent / "examples" / "block" / "scene.json"


def read_scene(manifest: Path):
    import numpy as np
    from PIL import Image

    spec = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(spec, dict):
        raise ValueError("The scene manifest must be a JSON object")
    for key in ("image", "queries", "instruction", "source"):
        if not isinstance(spec.get(key), str) or not spec[key].strip():
            raise ValueError(f"Expected a non-empty string for scene field {key!r}")
    if spec["source"] not in SOURCE_DOMAINS:
        raise ValueError(f"Unknown source {spec['source']!r}. Choose from {', '.join(SOURCE_DOMAINS)}")
    with Image.open(manifest.parent / spec["image"]) as im:
        rgb = np.array(im.convert("RGB"))
    with np.load(manifest.parent / spec["queries"], allow_pickle=False) as archive:
        if not {"points_xyz", "pixels_uv"}.issubset(archive.files):
            raise ValueError("The queries NPZ must contain points_xyz and pixels_uv")
        points, pixels = archive["points_xyz"], archive["pixels_uv"]
    return spec, rgb, points, pixels


def write_prediction(output: Path, result, pixels_uv, metadata: dict) -> None:
    import av
    import numpy as np

    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Choose a new output directory: {output}")
    with TemporaryDirectory(dir=output.parent, prefix=".worldflow-predict-") as tmp:
        stage = Path(tmp) / "prediction"
        stage.mkdir()
        frames = result["video"]
        with av.open(str(stage / "video.mp4"), mode="w") as container:
            stream = container.add_stream("libx264", rate=16)
            stream.width, stream.height = frames.shape[2], frames.shape[1]
            stream.pix_fmt = "yuv420p"
            for frame in frames:
                for packet in stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)

        source = result["points_xyz"]
        offsets = result["displacements"]
        selected = result["selected_indices"]
        np.savez_compressed(
            stage / "motion.npz",
            points_xyz=source,
            pixels_uv=pixels_uv[selected],
            selected_indices=selected,
            displacements=offsets,
            trajectories=np.concatenate([source[None], source[None] + offsets], axis=0),
        )
        (stage / "prediction.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        stage.rename(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path, help="Local WorldFlow inference bundle")
    parser.add_argument("--input", type=Path, default=DEFAULT_SCENE, help="Scene JSON (default: packaged blue block)")
    parser.add_argument("--output", type=Path, default=Path("outputs/block"), help="New result directory")
    parser.add_argument("--steps", type=int, default=40, help="Denoising steps (default: 40)")
    parser.add_argument("--guidance", type=float, default=2.0, help="Text guidance scale (default: 2.0)")
    parser.add_argument("--shift", type=float, default=5.0, help="Scheduler shift (default: 5.0)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        if args.output.exists():
            raise FileExistsError(f"Choose a new output directory: {args.output}")
        validate_bundle(args.checkpoint)
        spec, rgb, xyz, uv = read_scene(args.input)
        from .pipeline import predict

        result = predict(
            args.checkpoint,
            rgb,
            xyz,
            uv,
            spec["instruction"],
            spec["source"],
            steps=args.steps,
            guidance=args.guidance,
            shift=args.shift,
            seed=args.seed,
        )
        source = spec["source"]
        metadata = {
            "instruction": spec["instruction"],
            "source": source,
            "domain": SOURCE_DOMAINS[source],
            "flow_scale": DOMAIN_SCALES[SOURCE_DOMAINS[source]],
            "units": "meters",
            "coordinate_frame": "initial_camera",
            "displacement_reference": "first_frame",
            "seed": args.seed,
            "steps": args.steps,
            "guidance": args.guidance,
            "shift": args.shift,
            "fps": 16,
            "checkpoint": args.checkpoint.resolve().name,
        }
        write_prediction(args.output, result, uv, metadata)
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        parser.exit(2, f"worldflow: {exc}\n")
    print(f"Saved video.mp4, motion.npz and prediction.json in {args.output}")
