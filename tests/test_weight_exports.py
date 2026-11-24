"""One-path conversion, packaged model selection and legacy CLI compatibility."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed.checkpoint as dcp
from torch import nn

from worldflow.inference.loader import validate_bundle
from worldflow.inference.weights import load_weights
from worldflow.pack import main as pack_main
from worldflow.weights import dcp_iteration, inference_model_dir


class WeightExportTests(unittest.TestCase):
    def test_shared_export_retry_relocation_and_standalone_converter(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "weights"
            tokenizer = root / "tokenizer"
            tokenizer.mkdir(parents=True)
            for name in ("tokenizer.json", "tokenizer_config.json"):
                (tokenizer / name).write_text("{}")
            vae = root / "Wan2.2_VAE.pth"
            vae.touch()
            source = root / "imports/stage2/iter_000051100"
            expected = torch.arange(6, dtype=torch.float32).reshape(3, 2)
            dcp.save({"net_ema.llm2objectflow.weight": expected}, checkpoint_id=source / "model", no_dist=True)
            (source.parent / "iter_000051101").mkdir()
            self.assertEqual(dcp_iteration(source.parent, None), source)
            with self.assertRaises(FileNotFoundError):
                dcp_iteration(source.parent, 51101)
            with patch("sys.argv", ["worldflow-pack", "--checkpoint", str(root)]):
                pack_main()
            with (
                patch(
                    "sys.argv",
                    [
                        "worldflow-pack",
                        "--checkpoint",
                        str(root),
                        "--stage",
                        "2",
                        "--source",
                        "import",
                        "--iteration",
                        "51100",
                    ],
                ),
                patch("worldflow.pack._export_ema", side_effect=AssertionError("already converted")),
            ):
                pack_main()
            active = root / "models/stage2"
            self.assertTrue(active.is_symlink())
            self.assertFalse(active.readlink().is_absolute())

            legacy = Path(tmp) / "legacy"
            with patch(
                "sys.argv",
                [
                    "worldflow-pack",
                    "--dcp",
                    str(source),
                    "--tokenizer",
                    str(tokenizer),
                    "--vae",
                    str(vae),
                    "--output",
                    str(legacy),
                ],
            ):
                pack_main()
            moved = root.rename(Path(tmp) / "relocated")
            model = nn.Module()
            model.llm2objectflow = nn.Linear(2, 3, bias=False)
            for bin_path in (legacy, moved):
                validate_bundle(bin_path)
                load_weights(model, inference_model_dir(bin_path))
                torch.testing.assert_close(model.llm2objectflow.weight, expected)
            stamp = inference_model_dir(moved) / "worldflow-source.json"
            self.assertEqual(json.loads(stamp.read_text())["slot"], "stage2")
