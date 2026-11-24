"""Build the WorldFlow generator with FSDP2 and load each rank's checkpoint slices."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors import safe_open
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor
from torch.distributed.tensor._utils import compute_local_shape_and_global_offset

from ..models.network import WorldFlowNetwork


TRAINABLE_KEYS = ("moe_gen", "time_embedder", "vae2llm", "llm2vae", "objectflow")
STAGE_SCALES = {1: 0.0155919, 2: 0.01934810744226477}


def initialize_distributed():
    if not torch.cuda.is_available():
        raise RuntimeError("WorldFlow training requires CUDA with bfloat16 support")
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
    torch.cuda.set_device(device)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("WorldFlow training requires a GPU that supports bfloat16")
    if not dist.is_initialized():
        if "RANK" not in os.environ:
            # A direct single-rank invocation also uses the same FSDP2 code path.
            dist.init_process_group("nccl", store=dist.HashStore(), rank=0, world_size=1, device_id=device)
        else:
            dist.init_process_group("nccl", device_id=device)
    size = dist.get_world_size()
    local_size = int(os.environ.get("LOCAL_WORLD_SIZE", size))
    if size % local_size:
        raise ValueError("World size must be divisible by the number of local processes")
    if size > local_size:
        mesh = init_device_mesh("cuda", (size // local_size, local_size), mesh_dim_names=("replicate", "shard"))
    else:
        mesh = init_device_mesh("cuda", (size,), mesh_dim_names=("shard",))
    return device, mesh


def shard_network(network, mesh):
    """Place independent decoder blocks and the large frozen token tables in separate FSDP groups."""
    # The archived MoT checkpoints store both the live weights and the two
    # Adam moments in BF16. Only the separate power EMA uses FP32.
    network.to(dtype=torch.bfloat16)
    options = dict(mesh=mesh, reshard_after_forward=True)
    for name, param in network.named_parameters():
        param.requires_grad_(any(key in name for key in TRAINABLE_KEYS))
    backbone = network.language_model.model
    for index, layer in enumerate(backbone.layers):
        backbone.layers[index] = fully_shard(checkpoint_wrapper(layer, preserve_rng_state=False), **options)
    fully_shard(backbone.embed_tokens, **options)
    # The inference checkpoint includes this unused head. Keeping it separate
    # avoids all-gathering its vocabulary-sized weight on every forward.
    fully_shard(network.language_model.lm_head, **options)
    fully_shard(network, **options)
    return network


@torch.no_grad()
def load_sharded_weights(network, model_dir):
    """Read only this rank's contiguous dimensions from each indexed safetensors shard."""
    state = network.state_dict()
    index = json.loads((model_dir / "model.safetensors.index.json").read_text())["weight_map"]
    if state.keys() != index.keys():
        missing = sorted(state.keys() - index.keys())
        extra = sorted(index.keys() - state.keys())
        raise ValueError(f"Incompatible training bundle: missing {missing[:4]}, extra {extra[:4]}")
    files = defaultdict(list)
    for name, filename in index.items():
        if (
            not isinstance(filename, str)
            or (model_dir / filename).name != filename
            or not filename.endswith(".safetensors")
        ):
            raise ValueError(f"Invalid safetensors shard name {filename!r}")
        files[filename].append(name)
    for filename, names in files.items():
        with safe_open(model_dir / filename, framework="pt", device="cpu") as shard:
            if set(shard.keys()) != set(names):
                raise ValueError(f"Tensor names differ from the index in {filename}")
            for name in names:
                target = state[name]
                view = shard.get_slice(name)
                if tuple(view.get_shape()) != tuple(target.shape):
                    raise ValueError(f"Invalid checkpoint shape for {name}: {view.get_shape()} vs {target.shape}")
                if isinstance(target, DTensor):
                    shape, offset = compute_local_shape_and_global_offset(
                        target.shape, target.device_mesh, target.placements
                    )
                    slices = tuple(slice(start, start + length) for start, length in zip(offset, shape))
                    target.to_local().copy_(view[slices])
                else:
                    target.copy_(shard.get_tensor(name))


def build_network(checkpoint, stage, device, mesh, *, resume=False):
    from ..weights import training_model_dir

    initial_model = None
    if not resume:
        # Slots may be atomically republished. All ranks must read the same
        # immutable directory for the entire sharded initialization.
        chosen = [None]
        if not dist.is_initialized() or dist.get_rank() == 0:
            chosen[0] = str(training_model_dir(checkpoint, stage).resolve())
        if dist.is_initialized():
            dist.broadcast_object_list(chosen, src=0)
        initial_model = Path(chosen[0])
        if not (initial_model / "model.safetensors.index.json").is_file():
            raise FileNotFoundError(f"Missing stage {stage} initialization weights at {initial_model}")
    with torch.device("meta"):
        network = WorldFlowNetwork()
    network = shard_network(network, mesh)
    network.to_empty(device=device)
    network.language_model.model.rotary_emb.init_weights(buffer_device=device)
    if initial_model is not None:
        load_sharded_weights(network, initial_model)
        # A fresh stage adopts its own flow scale. A resume restores this
        # persistent buffer together with the live network and EMA from DCP.
        network.objectflow_scale_buf.fill_(STAGE_SCALES[stage])
    return network.train()
