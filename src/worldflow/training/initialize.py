"""Build a Stage 1 training bundle from a trusted Cosmos3-Nano video backbone."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from ..dcp import dcp_reader
from ..models.network import WorldFlowNetwork
from ..pack import write_model_shards
from ..weights import dcp_iteration, publish_model


def initialize_weights(dcp_dir: Path, destination: Path, *, seed: int) -> None:
    reader = dcp_reader(dcp_dir)
    tensors = reader.read_metadata().state_dict_metadata
    with torch.device("meta"):
        network = WorldFlowNetwork()
    state = network.state_dict()
    # Separate the new motion branch from the pretrained language and video modules.
    fresh = {k for k in state if k.startswith(("objectflow", "llm2objectflow."))}
    required = sorted(state.keys() - fresh)
    prefix = next((p for p in ("net_ema.", "net.", "") if p + "vae2llm.weight" in tensors), None)
    if prefix is None:
        raise ValueError("Could not find Cosmos3-Nano video projection weights in the DCP")
    if any(prefix + name in tensors for name in fresh):
        raise ValueError("This checkpoint already contains WorldFlow parameters. Export it with worldflow-pack.")
    source = {k: prefix + k for k in required}
    missing = [k for k, original in source.items() if original not in tensors]
    if missing:
        raise ValueError(f"Base checkpoint is missing {len(missing)} backbone tensors: {missing[:5]}")
    for name, original in source.items():
        if tensors[original].size != state[name].shape:
            raise ValueError(f"Incompatible backbone tensor {original}: expected {tuple(state[name].shape)}")

    # The backbone stays on meta. Only the new motion parameters are materialized here.
    network.objectflow2llm.to_empty(device="cpu")
    network.llm2objectflow.to_empty(device="cpu")
    network.objectflow_point_pe.to_empty(device="cpu")
    network.objectflow_modality_embed = torch.nn.Parameter(
        torch.empty_like(network.objectflow_modality_embed, device="cpu")
    )
    for name, buf in network.named_buffers(recurse=False):
        setattr(network, name, torch.empty_like(buf, device="cpu"))
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        network.initialize_objectflow()
    initialized = {k: v for k, v in network.state_dict().items() if k in fresh}
    print(f"Importing {len(source)} backbone weights and initializing {len(initialized)} WorldFlow tensors", flush=True)
    write_model_shards(reader, destination, source, initialized)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path, help="Shared weights directory with imports/base/")
    parser.add_argument("--iteration", type=int, help="Pretrained iteration in imports/base/ (default: latest)")
    parser.add_argument("--seed", type=int, default=42, help="Seed for new WorldFlow parameters")
    args = parser.parse_args()
    checkpoint = dcp_iteration(args.checkpoint / "imports/base", args.iteration)
    model = checkpoint if (checkpoint / ".metadata").is_file() else checkpoint / "model"
    version = checkpoint.name if checkpoint.name.startswith("iter_") else "checkpoint"
    publish_model(
        args.checkpoint,
        "base",
        f"{version}-seed{args.seed}",
        lambda target: initialize_weights(model, target, seed=args.seed),
        metadata=model / ".metadata",
    )


if __name__ == "__main__":
    main()
