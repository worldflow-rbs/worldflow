"""Resolve shared assets, model slots and DCP iterations beneath one local weights directory."""

from __future__ import annotations

import json
import os
import re
import errno
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory

_INDEX = "model.safetensors.index.json"
_ITERATION = re.compile(r"iter_(\d+)$")


def _legacy_stage(root: Path) -> int:
    info = root / "worldflow.json"
    if not info.is_file():
        return 2
    data = json.loads(info.read_text(encoding="utf-8"))
    stage = data.get("stage") if isinstance(data, dict) else None
    if type(stage) is not int or stage not in (1, 2):
        raise ValueError(f"Expected stage 1 or 2 in {info}")
    return stage


def inference_model_dir(root: Path, stage: int | None = None) -> tuple[Path, int]:
    root = Path(root)
    if stage is not None and (type(stage) is not int or stage not in (1, 2)):
        raise ValueError("Expected inference stage 1 or 2")
    if (root / "models").is_dir():
        stage = 2 if stage is None else stage
        model = root / "models" / f"stage{stage}"
    else:
        actual = _legacy_stage(root)
        if stage is not None and stage != actual:
            raise ValueError(f"The standalone checkpoint contains stage {actual}, requested stage {stage}")
        stage = actual
        model = root / "model"
    return model, stage


def training_model_dir(root: Path, stage: int) -> Path:
    root = Path(root)
    if stage not in (1, 2):
        raise ValueError("Expected training stage 1 or 2")
    if not (root / "models").is_dir() and (root / "model" / _INDEX).is_file():
        return root / "model"
    return root / "models" / ("base" if stage == 1 else "stage1")


def dcp_iteration(parent: Path, iteration: int | None = None) -> Path:
    """Resolve an imported DCP or the latest complete numbered iteration."""
    if iteration is not None and iteration < 0:
        raise ValueError("Expected a non-negative iteration")

    def has_metadata(path):
        return (path / ".metadata").is_file() or (path / "model" / ".metadata").is_file()

    if has_metadata(parent):
        if iteration is not None:
            raise ValueError(f"To select an iteration, place numbered iter_* directories under {parent}")
        return parent
    candidates = []
    for p in parent.glob("iter_*"):
        match = _ITERATION.fullmatch(p.name)
        if match and has_metadata(p) and (iteration is None or int(match[1]) == iteration):
            candidates.append((int(match[1]), p))
    if not candidates:
        which = "" if iteration is None else f" for iteration {iteration}"
        raise FileNotFoundError(f"No completed model DCP{which} under {parent}")
    return max(candidates)[1]


def publish_model(root: Path, slot: str, version: str, exporter, *, metadata: Path) -> None:
    """Build into a temporary directory and switch the active slot after all shards exist."""
    root = Path(root)
    if slot not in {"base", "stage1", "stage2"} or Path(version).name != version or version in {"", ".", ".."}:
        raise ValueError("Invalid WorldFlow model slot or version")
    for resource in ("tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json", "Wan2.2_VAE.pth"):
        if not (root / resource).is_file():
            raise FileNotFoundError(root / resource)
    slots = root / "models"
    target = slots / slot
    if target.exists() and not target.is_symlink():
        raise FileExistsError(f"Refusing to replace model directory {target}")
    slots.mkdir(exist_ok=True)
    if (root / "model" / _INDEX).is_file():
        previous = slots / f"stage{_legacy_stage(root)}"
        if not previous.exists() and not previous.is_symlink():
            try:
                previous.symlink_to("../model", target_is_directory=True)
            except FileExistsError:
                pass
    versions = root / "exports" / slot
    versions.mkdir(exist_ok=True, parents=True)
    destination = versions / version
    identity = {
        "slot": slot,
        "version": version,
        "dcp_metadata_sha256": hashlib.sha256(metadata.read_bytes()).hexdigest(),
    }
    with TemporaryDirectory(dir=root, prefix=".worldflow-export-") as temp:
        if not destination.exists():
            staging = Path(temp) / "model"
            exporter(staging)
            if not (staging / _INDEX).is_file():
                raise ValueError("Model export is missing its safetensors index")
            (staging / "worldflow-source.json").write_text(json.dumps(identity, indent=2) + "\n")
            try:
                staging.rename(destination)
            except OSError as error:
                if error.errno not in (errno.EEXIST, errno.ENOTEMPTY):
                    raise
        stamp = destination / "worldflow-source.json"
        if not (destination / _INDEX).is_file() or not stamp.is_file() or json.loads(stamp.read_text()) != identity:
            raise FileExistsError(f"A different model already occupies {destination}")
        link = Path(temp) / "active"
        link.symlink_to(os.path.relpath(destination, slots), target_is_directory=True)
        os.replace(link, target)
    print(f"Prepared {destination}. Active model: {target}", flush=True)
