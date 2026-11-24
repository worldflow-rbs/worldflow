"""Load the released WorldFlow Nano generator and its local inference assets."""

from pathlib import Path

from ..weights import inference_model_dir


def checkpoint_stage(checkpoint: Path, stage: int | None = None) -> int:
    """Resolve the prompt and motion-scale contract of the selected weights."""
    _, stage = inference_model_dir(Path(checkpoint), stage)
    return stage


def validate_shared_assets(checkpoint: Path) -> None:
    checkpoint = Path(checkpoint)
    for resource in (
        "tokenizer/tokenizer.json",
        "tokenizer/tokenizer_config.json",
        "Wan2.2_VAE.pth",
    ):
        if not (checkpoint / resource).is_file():
            raise FileNotFoundError(f"Missing {checkpoint / resource}. See checkpoint preparation in the README.")


def _resolve_bundle(checkpoint: Path, stage: int | None):
    validate_shared_assets(checkpoint)
    model_dir, resolved_stage = inference_model_dir(checkpoint, stage)
    # Pin one published model version while reading the index and successive shards.
    model_dir = model_dir.resolve()
    index = model_dir / "model.safetensors.index.json"
    if not index.is_file():
        raise FileNotFoundError(f"Missing WorldFlow model index: {index}")
    return model_dir, resolved_stage


def validate_bundle(checkpoint: Path, stage: int | None = None) -> None:
    _resolve_bundle(Path(checkpoint), stage)


def load_model(checkpoint: Path, stage: int | None = None):
    checkpoint = Path(checkpoint)
    model_dir, stage = _resolve_bundle(checkpoint, stage)
    import torch
    from transformers import AutoTokenizer

    from ..models.network import WorldFlowNetwork
    from ..models.vae import load_vae
    from .sampling import JointSampler
    from .weights import load_weights

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("WorldFlow Nano requires a CUDA GPU with bfloat16 support")

    device = torch.device("cuda", torch.cuda.current_device())
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tokenizer = AutoTokenizer.from_pretrained(checkpoint / "tokenizer", local_files_only=True)
    with torch.device("meta"):
        network = WorldFlowNetwork()
    network = network.to(dtype=torch.bfloat16).to_empty(device=device)
    # Rotary frequencies are generated values, so they are absent from the EMA index.
    network.language_model.model.rotary_emb.init_weights(buffer_device=device)
    load_weights(network, model_dir)
    vae = load_vae(checkpoint / "Wan2.2_VAE.pth", device=device)
    model = JointSampler(network, vae, tokenizer).eval()
    model.stage = stage
    return model
