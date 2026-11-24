"""Resolve and publish WorldFlow Nano EMA weights within a single local directory."""

from __future__ import annotations

import errno
import hashlib
import json
import os
import re
from pathlib import Path
from tempfile import TemporaryDirectory

INDEX = "model.safetensors.index.json"


def validate_shared_assets(root: Path) -> None:
    for name in ("tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json", "Wan2.2_VAE.pth"):
        if not (root / name).is_file():
            raise FileNotFoundError(f"Missing {root / name}. See checkpoint preparation in the README.")


def inference_model_dir(root: Path) -> Path:
    root = Path(root)
    if (root / "models").is_dir():
        model = root / "models" / "stage2"
    else:
        description = root / "worldflow.json"
        if description.is_file() and json.loads(description.read_text()).get("stage") != 2:
            raise ValueError("This inference release requires the Stage 2 WorldFlow Nano EMA model")
        model = root / "model"
    # Pin an export while the loader reads its index and successive shards.
    model = model.resolve()
    if not (model / INDEX).is_file():
        raise FileNotFoundError(f"Missing WorldFlow model index: {model / INDEX}")
    return model


def dcp_iteration(parent: Path, iteration: int | None) -> Path:
    if iteration is not None and iteration < 0:
        raise ValueError("Expected a non-negative iteration")

    def has_metadata(path):
        return (path / ".metadata").is_file() or (path / "model" / ".metadata").is_file()

    if has_metadata(parent):
        if iteration is not None:
            raise ValueError(f"To select an iteration, place numbered iter_* directories under {parent}")
        return parent
    candidates = []
    for path in parent.glob("iter_*"):
        match = re.fullmatch(r"iter_(\d+)", path.name)
        if match and has_metadata(path) and (iteration is None or int(match[1]) == iteration):
            candidates.append((int(match[1]), path))
    if not candidates:
        which = "" if iteration is None else f" for iteration {iteration}"
        raise FileNotFoundError(f"No completed model DCP{which} under {parent}")
    return max(candidates)[1]


def publish_ema(root: Path, iteration: int | None, exporter) -> None:
    """Publish a complete export and atomically select it for the next inference call."""
    validate_shared_assets(root)
    checkpoint = dcp_iteration(root / "imports" / "stage2", iteration)
    model = checkpoint if (checkpoint / ".metadata").is_file() else checkpoint / "model"
    stage_info = model.parent / "training.json"
    if stage_info.is_file() and json.loads(stage_info.read_text())["identity"]["stage"] != 2:
        raise ValueError("This checkpoint is not a Stage 2 WorldFlow EMA")
    version = "import-" + (checkpoint.name if checkpoint.name.startswith("iter_") else "checkpoint")
    destination = root / "exports" / "stage2" / version
    slots = root / "models"
    target = slots / "stage2"
    if target.exists() and not target.is_symlink():
        raise FileExistsError(f"Refusing to replace model directory {target}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    identity = {
        "slot": "stage2",
        "version": version,
        "dcp_metadata_sha256": hashlib.sha256((model / ".metadata").read_bytes()).hexdigest(),
    }
    with TemporaryDirectory(dir=root, prefix=".worldflow-export-") as temp:
        if not destination.exists():
            staging = Path(temp) / "model"
            exporter(model, staging)
            if not (staging / INDEX).is_file():
                raise ValueError("Model export is missing its safetensors index")
            (staging / "worldflow-source.json").write_text(json.dumps(identity, indent=2) + "\n")
            try:
                staging.rename(destination)
            except OSError as error:
                if error.errno not in (errno.EEXIST, errno.ENOTEMPTY):
                    raise
        stamp = destination / "worldflow-source.json"
        if not (destination / INDEX).is_file() or not stamp.is_file() or json.loads(stamp.read_text()) != identity:
            raise FileExistsError(f"A different model already occupies {destination}")
        slots.mkdir(exist_ok=True)
        link = Path(temp) / "active"
        link.symlink_to(os.path.relpath(destination, slots), target_is_directory=True)
        os.replace(link, target)
    print(f"Prepared {destination}. Active model: {target}", flush=True)
