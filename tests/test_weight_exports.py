"""Verify shared, relocatable weight slots and safe reactivation of an existing export."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from worldflow.weights import dcp_iteration, inference_model_dir, publish_model, training_model_dir


class WeightExportTests(unittest.TestCase):
    def test_selection_and_retry_under_one_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "weights"
            (root / "tokenizer").mkdir(parents=True)
            for name in ("tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json", "Wan2.2_VAE.pth"):
                (root / name).touch()
            source = root / "runs/stage1/iter_000032000/model"
            source.mkdir(parents=True)
            metadata = source / ".metadata"
            metadata.write_bytes(b"saved DCP identity")
            self.assertEqual(dcp_iteration(root / "runs/stage1", 32000), source.parent)

            def write_model(destination):
                destination.mkdir()
                (destination / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {}}))

            writer = Mock(side_effect=write_model)
            for _ in range(2):
                publish_model(root, "stage1", "run-iter_000032000", writer, metadata=metadata)
            writer.assert_called_once()
            moved = root.rename(Path(temporary) / "relocated")
            model, stage = inference_model_dir(moved, 1)
            self.assertEqual(stage, 1)
            self.assertTrue((model / "model.safetensors.index.json").is_file())
            self.assertEqual(training_model_dir(moved, 2), model)
            self.assertEqual(inference_model_dir(moved), (moved / "models/stage2", 2))
            metadata = moved / metadata.relative_to(root)
            metadata.write_bytes(b"different DCP, same iteration")
            with self.assertRaises(FileExistsError):
                publish_model(moved, "stage1", "run-iter_000032000", writer, metadata=metadata)
