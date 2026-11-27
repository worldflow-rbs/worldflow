"""CPU regressions for training schedules and optimizer/EMA checkpoint restoration."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
from torch import nn

from worldflow.training.checkpoint import PowerEMA, restore_checkpoint, save_checkpoint
from worldflow.training.cli import main as train_main
from worldflow.training.distributed import build_network
from worldflow.training.engine import TrainConfig, lr_multiplier


class TrainingEngineTests(unittest.TestCase):
    def test_training_cli_resolves_run_and_resume_paths(self):
        root, manifest = Path("weights"), Path("clips.jsonl")
        with patch("worldflow.training.engine.train") as train:
            for stage in (1, 2):
                with patch(
                    "sys.argv",
                    [
                        "worldflow-train",
                        "--stage",
                        str(stage),
                        "--checkpoint",
                        str(root),
                        "--manifest",
                        str(manifest),
                        "--resume-step",
                        "32",
                    ],
                ):
                    train_main()
                config = train.call_args.args[0]
                self.assertEqual(config.run_dir, root / "runs" / f"stage{stage}")
                self.assertEqual(config.resume_dir, config.run_dir / "iter_000000032")
                config.validate()
        custom = TrainConfig(2, root, manifest, output=Path("custom"), resume_step=7)
        self.assertEqual(custom.resume_dir, Path("custom/iter_000000007"))
        explicit = TrainConfig(1, root, manifest, output=Path("custom"), resume=Path("some/iter_1"))
        self.assertEqual(explicit.resume_dir, Path("some/iter_1"))
        with self.assertRaisesRegex(ValueError, "either --resume"):
            TrainConfig(2, root, manifest, resume=Path("ckpt"), resume_step=7).validate()

    def test_resume_never_reads_initialization_weights(self):
        # The live tensors are restored from DCP after model construction. The
        # original stage slot can already have been moved or deleted.
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch("worldflow.training.distributed.WorldFlowNetwork"),
            patch("worldflow.training.distributed.shard_network", return_value=MagicMock()),
            patch("worldflow.training.distributed.load_sharded_weights") as load,
            patch("worldflow.weights.training_model_dir", side_effect=AssertionError("unexpected init lookup")),
        ):
            build_network(Path(tmp), 2, torch.device("cpu"), None, resume=True)
            load.assert_not_called()

    def test_fresh_stage_loads_the_preceding_slot(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch("worldflow.training.distributed.WorldFlowNetwork"),
            patch("worldflow.training.distributed.shard_network", return_value=MagicMock()),
            patch("worldflow.training.distributed.load_sharded_weights") as load,
        ):
            root = Path(tmp)
            for stage, slot in ((1, "base"), (2, "stage1")):
                model = root / "models" / slot
                model.mkdir(parents=True)
                (model / "model.safetensors.index.json").write_text("{}")
                build_network(root, stage, torch.device("cpu"), None)
                self.assertEqual(load.call_args.args[1], model.resolve())

    def test_paper_optimizer_schedules(self):
        self.assertEqual(lr_multiplier(1, 0, 32350, 500), 0)
        self.assertEqual(lr_multiplier(1, 500, 32350, 500), 1)
        self.assertAlmostEqual(lr_multiplier(1, (38820 + 500) // 2, 32350, 500), 0.5)
        for step, ratio in ((250, 0.5), (11500, 1), (12500, 0.8), (42100, 0.8), (48100, 0.2), (51100, 0.05)):
            self.assertAlmostEqual(lr_multiplier(2, step, 51100, 500), ratio)
        config = TrainConfig(1, Path("bundle"), Path("data"), Path("run"), stop_after=32000)
        self.assertEqual(config.total_steps, 32350)
        config.validate()

    def test_resume_restores_adam_ema_and_frozen_parameters(self):
        def init():
            torch.manual_seed(4)
            model = nn.Sequential(nn.Linear(3, 3), nn.Linear(3, 2))
            model[0].requires_grad_(False)
            optimizer = torch.optim.AdamW(model[1].parameters(), lr=0.01)
            return model, optimizer, PowerEMA(model)

        def update(model, optimizer, ema, step):
            model(torch.ones(1, 3)).square().sum().backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            ema.update(step)

        with tempfile.TemporaryDirectory() as tmp:
            net, opt, ema = init()
            update(net, opt, ema, 0)
            folder = save_checkpoint(tmp, net, ema, opt, 1, {"stage": 1})
            update(net, opt, ema, 1)
            expected = {name: tensor.clone() for name, tensor in ema.model_state().items()}
            net, opt, ema = init()
            with self.assertRaisesRegex(ValueError, "stage"):
                restore_checkpoint(folder, net, ema, opt, {"stage": 2})
            self.assertEqual(restore_checkpoint(folder, net, ema, opt, {"stage": 1}), 1)
            update(net, opt, ema, 1)
            for key, value in ema.model_state().items():
                torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
