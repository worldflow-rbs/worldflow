# Local weights directory

Set one path for all WorldFlow commands.

```bash
export WEIGHTS="$PWD/checkpoints/worldflow-nano"
worldflow --checkpoint "$WEIGHTS"
```

For inference, a prepared directory contains the Stage 2 model, the tokenizer and one copy of the Wan 2.2 VAE. Install the package and keep these files together for offline generation. The same directory can also hold the base model, Stage 1 generator and resumable training checkpoints.

```text
worldflow-nano/
├── tokenizer/                # Cosmos3-Nano tokenizer files
├── Wan2.2_VAE.pth            # shared video autoencoder
├── models/
│   ├── base                  # Cosmos3-Nano with new WorldFlow motion parameters
│   ├── stage1                # active warm-up EMA
│   └── stage2                # active final EMA, default for inference
├── exports/                  # versioned model shards selected by models/
├── runs/                     # resumable DCPs written by worldflow-train
│   ├── stage1/iter_000032000/
│   └── stage2/iter_000051100/
└── imports/                  # optional source DCPs for conversion
    ├── base/iter_000048464/
    ├── stage1/iter_000032000/
    └── stage2/iter_000051100/
```

Each active model resolves to a directory containing `model.safetensors.index.json` and its shards. Export commands create relative symlinks under `models/` and store each converted version under `exports/`. Copy the directory with links preserved, or copy the resolved active shards into the corresponding `models/stage*` directory on an inference machine. Serving a model requires only its active weights, the tokenizer and VAE. A standalone release bundle with `model/` at the top level also works.

| Command | What it reads and writes under `$WEIGHTS` |
| --- | --- |
| `worldflow-init --checkpoint "$WEIGHTS"` | Reads the trusted Cosmos3-Nano DCP under `imports/base/`. Creates the `models/base` training initialization. |
| `worldflow-train --checkpoint "$WEIGHTS" --stage 1 --manifest data/stage1.jsonl` | Reads `models/base`, writes resumable iterations under `runs/stage1/`. |
| `worldflow-pack --checkpoint "$WEIGHTS" --stage 1 --iteration 32000` | Exports that iteration's EMA and selects it as `models/stage1`. |
| `worldflow-train --checkpoint "$WEIGHTS" --stage 2 --manifest data/stage2.jsonl` | Reads `models/stage1`, writes under `runs/stage2/`. Add `--resume-step N` to continue the same training run. |
| `worldflow-pack --checkpoint "$WEIGHTS" --stage 2 --iteration 51100` | Exports the Stage 2 EMA and selects it as `models/stage2`. |
| `worldflow --checkpoint "$WEIGHTS"` | Generates from the active Stage 2 model. Add `--stage 1` to use the warm-up model. |
| `worldflow-evaluate --checkpoint "$WEIGHTS" --manifest data/heldout.jsonl --output outputs/heldout` | Scores the selected generator using the shared assets. |

If you already have the paper training DCP, place its iteration under `imports/stage2/` with the model shards and `model/.metadata` intact. Then run `worldflow-pack --checkpoint "$WEIGHTS" --source import --stage 2 --iteration 51100`. The converter prefers completed iterations in `runs/` by default, and uses `imports/` when a training run is absent. Without `--iteration` it selects the largest completed iteration number. A converted full EMA takes about 57 GiB plus the shared VAE and tokenizer.
