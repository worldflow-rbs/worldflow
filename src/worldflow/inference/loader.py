"""Load the released WorldFlow Nano generator and its local inference assets."""

from pathlib import Path

from ..weights import inference_model_dir, validate_shared_assets


def validate_bundle(checkpoint: Path) -> None:
    validate_shared_assets(checkpoint)
    inference_model_dir(checkpoint)


def load_model(checkpoint: Path):
    checkpoint = Path(checkpoint)
    validate_shared_assets(checkpoint)
    model_dir = inference_model_dir(checkpoint)
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
    return JointSampler(network, vae, tokenizer).eval()
