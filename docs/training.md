# Train WorldFlow Nano

The paper recipe has two optimizer runs. Start Stage 1 from a Cosmos3-Nano video-language backbone. Export the Stage 1 EMA and use it to initialize Stage 2. Both stages use the same model and package as inference. Training requires Linux, PyTorch 2.8.0, CUDA GPUs with BF16 support, a local tokenizer and the Wan 2.2 video VAE. Follow the installation steps in the [README](../README.md).

## Set the weights directory and build manifests

Install the data-preparation version of NumPy with `python -m pip install '.[training]'` and make the FFmpeg executable available for native Real3 data. NumPy 1.26.4 reproduces the original point-selection ordering for tied voxel candidates.

Set `WEIGHTS` once. Place the local tokenizer in `$WEIGHTS/tokenizer/`, the shared video VAE at `$WEIGHTS/Wan2.2_VAE.pth`, and the original pretrained iteration under `$WEIGHTS/imports/base/`. The [weights directory layout](weights.md) is shared by the trainer, exporter, evaluation and inference commands. The importer reads the pretrained video and language EMA and initializes the new 3D input projection, output head, modality embedding and geometry encoder. The 3D output projection starts at zero.

```bash
export WEIGHTS="$PWD/checkpoints/worldflow-nano"
worldflow-init --checkpoint "$WEIGHTS"

worldflow-prepare --root /datasets/worldflow --stage 1 --output data/stage1.jsonl
worldflow-prepare --root /datasets/worldflow --stage 2 --output data/stage2.jsonl
```

The [data guide](data.md) describes the source layouts, split and exclusion files, and portable training samples. Each example supplies the full 49-frame clip, the instruction, the 64 initial 3D points and their source pixels, and 48 frame-to-frame displacement targets in meters.

## Train and export each stage

The commands below run on one eight-GPU node. The reference experiment used eight nodes with eight ranks each, two examples per rank, and one microbatch per update (global batch 128). For multiple nodes, launch on each node with its own `--node-rank`, a common `--master-addr` and `--master-port`, `--nnodes=8` and `--nproc-per-node=8` in place of `--standalone --nproc-per-node=8`. Use shared paths for the data, initialization bundle and run directory. On fewer GPUs, `--grad-accum` controls the effective batch size.

```bash
torchrun --standalone --nproc-per-node=8 -m worldflow.training.cli \
  --stage 1 --checkpoint "$WEIGHTS" --manifest data/stage1.jsonl --stop-after 32000

worldflow-pack --checkpoint "$WEIGHTS" --stage 1 --iteration 32000

torchrun --standalone --nproc-per-node=8 -m worldflow.training.cli \
  --stage 2 --checkpoint "$WEIGHTS" --manifest data/stage2.jsonl

worldflow-pack --checkpoint "$WEIGHTS" --stage 2 --iteration 51100

worldflow --checkpoint "$WEIGHTS" --output outputs/trained-block
```

Stage 1 uses the archived 32,350-step horizon and a 38,820-step cosine cycle. The paper takes the EMA at update 32,000. `--stop-after` preserves the original schedule. Stage 2 starts a fresh AdamW optimizer and EMA from the active Stage 1 export, sets the new motion scale, and runs for 51,100 updates. Inference defaults to the active Stage 2 export. Select the warm-up model with `worldflow --checkpoint "$WEIGHTS" --stage 1`.

## Training recipe

| Setting | Stage 1 | Stage 2 |
| --- | --- | --- |
| Data | SpatialCog / Demo5 | Sim7, Real3, Demo5 and Molmo-Motion |
| Text | Chinese instruction with cleaning and paraphrase augmentation | SPV1 instruction and scene context |
| Probability of dropping all text | 0 | 0.1 |
| Motion normalization | 0.0155919 m per component | Per-domain meter scales in `inference/presets.py` |
| Direction / velocity / endpoint auxiliary weights | 0.15 / 0.015 / 0.02 | 0.1 / 0.1 / 0.05 |
| Learning rate | Warm-up then cosine | Warm-up then phase-dependent cosine transitions |

The frozen VAE encodes 49 RGB frames at 480 × 288. The first latent frame conditions the network. The joint objective predicts video and 3D flow velocities at a shared noise level. The trajectory terms compare cumulative direction, step displacement and final displacement. Training updates the generator expert, video and motion projections, time embedding and motion geometry modules. The reasoner and VAE are frozen.

Both stages use AdamW with peak LR `1e-4`, betas `(0.9, 0.95)`, epsilon `1e-6`, zero weight decay, 500 warm-up steps, gradient clipping at 0.1 and a power-function EMA. The distributed trainer uses PyTorch FSDP2 and decoder-block activation checkpointing. Live model weights and Adam moments use BF16. EMA weights use FP32. Within a node the ranks shard the network. Across nodes the local sharding groups are replicated.

The second-stage sampler moves through five phases of 12,000, 12,000, 100, 24,000 and 3,000 updates. It blends adjacent sampling probabilities over 1,000, 50, 50 and 1,000 updates. Within a domain it weights sources by the square root of their valid example count, or by count to the power of 0.8 for Real3. The indexed sampler draws with replacement and derives draws from the rank and microbatch position so a resumed run continues its sequence. Use `--sampling uniform` for a custom subset or a smoke run.

## Resume a run

Each completed iteration under `$WEIGHTS/runs/stage1/` or `$WEIGHTS/runs/stage2/` contains model and EMA tensors, sharded optimizer state and training metadata. To resume the Stage 2 command above, add `--resume-step 2000` with the same recipe, data, tokenizer, VAE and number of ranks. Use `--stop-after` to pause after a chosen update. `--steps` changes the schedule horizon.

For a two-update installation check, use a small manifest with `--steps 2 --warmup-steps 0 --batch-size 1 --sampling uniform --stop-after 1`. Resume the resulting `iter_000000001` with the same arguments and `--stop-after 2`. Export the EMA to run generation, or evaluate held-out clips with the [evaluation CLI](evaluation.md).
