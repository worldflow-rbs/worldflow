"""Prepare a portable inference bundle from a trusted WorldFlow Nano EMA checkpoint."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "merges.txt",
    "vocab.json",
    "chat_template.json",
    "chat_template.jinja",
    "special_tokens_map.json",
    "added_tokens.json",
)


def _export_ema(dcp_dir: Path, destination: Path) -> None:
    import torch
    import torch.distributed.checkpoint as dcp
    from safetensors.torch import save_file

    reader = dcp.FileSystemReader(dcp_dir)
    tensors = reader.read_metadata().state_dict_metadata
    keys = sorted(key for key in tensors if key.startswith("net_ema."))
    if "net_ema.llm2objectflow.weight" not in keys:
        raise ValueError("Expected a WorldFlow DCP model directory with EMA object-flow weights")
    shards = []
    current = []
    total_bytes = current_bytes = 0
    for key in keys:
        meta = tensors[key]
        size = meta.size.numel() * meta.properties.dtype.itemsize
        if current and current_bytes + size > 2 * 1024**3:
            shards.append(current)
            current, current_bytes = [], 0
        current.append(key)
        current_bytes += size
        total_bytes += size
    shards.append(current)

    destination.mkdir()
    weight_map = {}
    for number, names in enumerate(shards, start=1):
        filename = f"model-{number:05d}-of-{len(shards):05d}.safetensors"
        print(f"Exporting {filename}", flush=True)
        state = {k: torch.empty(tensors[k].size, dtype=tensors[k].properties.dtype) for k in names}
        dcp.load(state, storage_reader=reader, no_dist=True)
        flat = {k.removeprefix("net_ema."): value for k, value in state.items()}
        save_file(flat, destination / filename)
        weight_map.update(dict.fromkeys(flat, filename))
        del state, flat
    index = {"metadata": {"total_size": total_bytes}, "weight_map": weight_map}
    (destination / "model.safetensors.index.json").write_text(json.dumps(index, indent=2) + "\n")


def prepare_bundle(dcp_path: Path, tokenizer: Path, vae: Path, output: Path) -> None:
    model = dcp_path if (dcp_path / ".metadata").is_file() else dcp_path / "model"
    if not (model / ".metadata").is_file():
        raise ValueError("Expected an iteration or model directory containing the DCP model metadata")
    for name in ("tokenizer.json", "tokenizer_config.json"):
        if not (tokenizer / name).is_file():
            raise FileNotFoundError(tokenizer / name)
    if not vae.is_file():
        raise FileNotFoundError(vae)
    if output.exists():
        raise FileExistsError(f"Choose a new output directory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix=".worldflow-pack-") as tmp:
        bundle = Path(tmp) / "bundle"
        (bundle / "tokenizer").mkdir(parents=True)
        for name in TOKENIZER_FILES:
            if (tokenizer / name).is_file():
                shutil.copyfile(tokenizer / name, bundle / "tokenizer" / name)
        shutil.copyfile(vae, bundle / "Wan2.2_VAE.pth")
        _export_ema(model, bundle / "model")
        bundle.rename(output)
    print(f"Prepared {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dcp", required=True, type=Path, help="Trusted WorldFlow iteration or model directory")
    parser.add_argument("--tokenizer", required=True, type=Path, help="Local Cosmos3-Nano tokenizer directory")
    parser.add_argument("--vae", required=True, type=Path, help="Local Wan2.2_VAE.pth file")
    parser.add_argument("--output", required=True, type=Path, help="New inference bundle directory")
    args = parser.parse_args()
    prepare_bundle(args.dcp, args.tokenizer, args.vae, args.output)


if __name__ == "__main__":
    main()
