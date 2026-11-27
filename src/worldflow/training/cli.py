"""Train the WorldFlow generator from a local weights directory."""

import argparse
from pathlib import Path


def main():
    from .engine import STAGE_STEPS, TrainConfig, train

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=int, choices=(1, 2), required=True)
    parser.add_argument(
        "--checkpoint", type=Path, required=True, help="Weights root containing shared tokenizer, VAE and stage models"
    )
    parser.add_argument("--manifest", type=Path, required=True, help="Training JSONL manifest")
    parser.add_argument("--output", type=Path, help="Run directory (default: ROOT/runs/stage1 or ROOT/runs/stage2)")
    resume = parser.add_mutually_exclusive_group()
    resume.add_argument(
        "--resume", type=Path, help="Resume network, EMA, optimizer and data cursor from this iteration"
    )
    resume.add_argument("--resume-step", type=int, help="Resume iteration N from the selected run directory")
    parser.add_argument(
        "--steps", type=int, help=f"Curriculum and LR horizon (stage 1: {STAGE_STEPS[1]}, stage 2: {STAGE_STEPS[2]})"
    )
    parser.add_argument("--stop-after", type=int, help="Stop and save at this step while preserving the full schedule")
    parser.add_argument("--batch-size", type=int, default=2, help="Samples per rank per microbatch (default: 2)")
    parser.add_argument("--grad-accum", type=int, default=1, help="Microbatches per optimizer step (default: 1)")
    parser.add_argument("--sampling", choices=("curriculum", "uniform"), default="curriculum")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--save-every", type=int, default=2000)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--workers", type=int, default=0, help="DataLoader worker processes per rank")
    parser.add_argument("--seed", type=int, default=42)
    train(TrainConfig(**vars(parser.parse_args())))


if __name__ == "__main__":
    main()
