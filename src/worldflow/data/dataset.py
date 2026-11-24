"""Random access to portable JSONL training manifests."""

import json
from array import array
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from worldflow.inference.presets import SOURCE_DOMAINS
from .common import FRAMES, HEIGHT, WIDTH, POINTS


def collate_training_samples(samples):
    """Preserve individual instructions and per-sample geometry for the training objective."""
    return list(samples)


class TrainingDataset(Dataset):
    """Each row identifies a single object clip. Store offsets instead of loading every path into RAM."""

    def __init__(self, manifest):
        self.manifest = Path(manifest).resolve()
        self.offsets = array("Q")
        self.source_indices = defaultdict(list)
        with self.manifest.open("rb") as stream:
            while True:
                offset = stream.tell()
                line = stream.readline()
                if not line:
                    break
                if not line.strip():
                    continue
                row = json.loads(line)
                source = row["source"]
                if source not in SOURCE_DOMAINS or not isinstance(row["sample_id"], str) or not row["sample_id"]:
                    raise ValueError(f"invalid training manifest source or sample_id at offset {offset}")
                self.source_indices[source].append(len(self.offsets))
                self.offsets.append(offset)
        if not self.offsets:
            raise ValueError(f"training manifest is empty: {self.manifest}")

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, index):
        with self.manifest.open("rb") as stream:
            stream.seek(self.offsets[index])
            row = json.loads(stream.readline())
        return read_sample(row, self.manifest.parent)


def read_sample(row, base):
    from .adapters import load_native

    kind = row.get("format", "prepared")
    if kind == "prepared":
        with np.load(base / row["sample"], allow_pickle=False) as source:
            sample = {
                name: torch.from_numpy(source[name]) for name in ("video", "flow_m", "points_xyz", "pixel_indices")
            }
    else:
        sample = load_native(kind, row, base)
    sample.update(source=row["source"], sample_id=row["sample_id"])
    sample["instruction"] = row.get("instruction") or sample.get("instruction", "")
    if row["source"] == "demo5" and isinstance(sample["instruction"], str):
        from worldflow.training.augmentation import clean_instruction

        sample["instruction"] = clean_instruction(sample["instruction"])
    if not isinstance(sample["instruction"], str) or not sample["instruction"].strip():
        raise ValueError(f"{row['sample_id']}: missing instruction")
    if "prompt_context" in row:
        sample["prompt_context"] = row["prompt_context"]
    if "source_index" in row:
        value = row["source_index"]
        if type(value) is not int or value < 0:
            raise ValueError("source_index must be a nonnegative integer")
        sample["source_index"] = value
    if "instruction_variants" in row:
        variants = row["instruction_variants"]
        if not isinstance(variants, list) or any(not isinstance(v, str) or not v.strip() for v in variants):
            raise ValueError("instruction_variants must be a list of nonempty alternate instructions")
        sample["instruction_variants"] = variants
    for name, shape, dtype in (
        ("video", (FRAMES, HEIGHT, WIDTH, 3), torch.uint8),
        ("flow_m", (FRAMES - 1, POINTS, 3), torch.float32),
        ("points_xyz", (POINTS, 3), torch.float32),
        ("pixel_indices", (POINTS,), torch.int64),
    ):
        tensor = sample[name]
        if tuple(tensor.shape) != shape or tensor.dtype != dtype:
            raise ValueError(f"{row['sample_id']}: invalid {name}, expected {shape} {dtype}")
        if tensor.is_floating_point() and not torch.isfinite(tensor).all():
            raise ValueError(f"{row['sample_id']}: non-finite {name}")
    pixels = sample["pixel_indices"]
    if (pixels < 0).any() or (pixels >= HEIGHT * WIDTH).any():
        raise ValueError(f"{row['sample_id']}: source pixel outside training image")
    return sample
