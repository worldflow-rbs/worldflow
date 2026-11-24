"""Regression checks for first-frame-only held-out inference and paper trajectory metrics."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from worldflow.evaluation.cli import evaluate
from worldflow.evaluation.metrics import trajectory_metrics


class EvaluationTests(unittest.TestCase):
    def test_inclusive_thresholds_and_temporal_integration(self):
        increments = np.array([[[0.01, 0, 0]], [[0.01, 0, 0]]], dtype=np.float32)
        metrics = trajectory_metrics(np.zeros_like(increments), increments)
        self.assertAlmostEqual(metrics["ade"], 0.015)
        self.assertAlmostEqual(metrics["fde"], 0.02)
        self.assertEqual(metrics["pwt_1cm"], 0.5)
        self.assertEqual(metrics["pwt_2cm"], 1)
        self.assertEqual(metrics["pwt_5cm"], 1)
        self.assertAlmostEqual(metrics["pwt"], 2.5 / 3)
        self.assertEqual(trajectory_metrics(increments.cumsum(0), increments)["ade"], 0)

    def test_manifest_scoring_uses_only_frame_zero_and_loads_once(self):
        model = object()
        observed = np.zeros((4, 5, 3), dtype=np.uint8)
        future = np.full_like(observed, 255)
        common = {
            "video": np.stack((observed, future, future)),
            "points_xyz": np.zeros((2, 3)),
            "pixel_indices": np.array([0, 14]),
            "instruction": "move",
            "flow_m": np.zeros((2, 2, 3)),
        }
        samples = [
            dict(common, source="sim7", sample_id="a"),
            dict(common, source="sim7", sample_id="b"),
            dict(common, source="demo5", sample_id="c", source_index=8),
        ]

        def prediction(loaded, frame, points, uv, instruction, source, **options):
            self.assertIs(loaded, model)
            np.testing.assert_array_equal(frame, observed)
            np.testing.assert_array_equal(uv, [[0, 0], [4, 2]])
            return {"displacements": np.zeros((2, 2, 3)), "selected_indices": [0, 1]}

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "scores"
            with (
                patch("worldflow.evaluation.cli.TrainingDataset", return_value=samples),
                patch("worldflow.evaluation.cli.checkpoint_stage", return_value=2),
                patch("worldflow.evaluation.cli.load_model", return_value=model) as loader,
                patch("worldflow.evaluation.cli.predict_with_model", side_effect=prediction),
            ):
                summary = evaluate(Path("bundle"), Path("test.jsonl"), output)
            loader.assert_called_once()
            records = [json.loads(row) for row in (output / "samples.jsonl").read_text().splitlines()]
            self.assertEqual([row["seed"] for row in records], [42, 43, 50])
            self.assertEqual(summary["overall"]["pwt"], 1)
            self.assertEqual(summary["by_source"]["sim7"]["samples"], 2)
            self.assertEqual(summary["by_source"]["demo5"]["samples"], 1)
