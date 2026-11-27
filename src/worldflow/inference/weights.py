"""Load the EMA generator weights from a local sharded safetensors bundle."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open


@torch.no_grad()
def load_weights(network: torch.nn.Module, model_dir: Path) -> None:
    """Load every parameter and persistent buffer, validating the complete key set."""
    index = json.loads((model_dir / "model.safetensors.index.json").read_text())
    weights = index["weight_map"]
    state = network.state_dict()
    missing = state.keys() - weights.keys()
    extra = weights.keys() - state.keys()
    if missing or extra:
        raise ValueError(
            f"Incompatible WorldFlow weights. Missing {len(missing)} tensors {sorted(missing)[:5]}, "
            f"unexpected {len(extra)} tensors {sorted(extra)[:5]}."
        )

    shards = defaultdict(set)
    for key, filename in weights.items():
        if Path(filename).name != filename or not filename.endswith(".safetensors"):
            raise ValueError(f"Invalid shard name {filename!r}")
        shards[filename].add(key)

    for filename, keys in shards.items():
        with safe_open(model_dir / filename, framework="pt", device="cpu") as shard:
            if set(shard.keys()) != keys:
                raise ValueError(f"Tensor names in {filename} disagree with the checkpoint index")
            for key in sorted(keys):
                tensor = shard.get_tensor(key)
                if tensor.shape != state[key].shape:
                    raise ValueError(f"Invalid shape for {key}: {tensor.shape}, expected {state[key].shape}")
                if state[key].is_meta:
                    raise RuntimeError(f"Cannot copy weights into unallocated parameter {key}")
                state[key].copy_(tensor)
