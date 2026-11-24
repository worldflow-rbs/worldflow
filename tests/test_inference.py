"""CPU regression checks for portable assets, checkpoint selection and trajectory convention."""

import json
import tempfile
import unittest
from pathlib import Path

import av
import numpy as np
import torch
from torch import nn
from torch.distributed import checkpoint as dcp

from worldflow.inference.cli import DEFAULT_SCENE, read_scene, write_prediction
from worldflow.inference.pipeline import _select_points
from worldflow.inference.weights import load_weights
from worldflow.pack import _export_ema


class InferenceTests(unittest.TestCase):
    def test_bundled_inputs_and_query_correspondence(self):
        scene, rgb, points, pixels = read_scene(DEFAULT_SCENE)
        self.assertEqual(scene["source"], "sim7")
        self.assertEqual(rgb.shape, (288, 480, 3))
        selected, flat, rows = _select_points(rgb, points, pixels)
        np.testing.assert_array_equal(selected, points)
        np.testing.assert_array_equal(rows, np.arange(64))
        np.testing.assert_array_equal(flat, pixels[:, 1] * 480 + pixels[:, 0])
        selected, _, rows = _select_points(rgb, points[:3], pixels[:3])
        np.testing.assert_array_equal(selected, points[:3][rows])
        self.assertEqual(len(rows), 64)

    def test_output_frame_zero_and_metric_offsets(self):
        with tempfile.TemporaryDirectory() as work:
            folder = Path(work) / "results"
            initial = np.array([[0.5, 0.25, 1.0]], dtype=np.float32)
            offsets = np.array([[[0.0, 0.1, 0.0]], [[0.0, 0.15, 0.0]]], dtype=np.float32)
            result = {
                "video": np.zeros((3, 16, 16, 3), dtype=np.uint8),
                "points_xyz": initial,
                "selected_indices": np.array([1]),
                "displacements": offsets,
            }
            write_prediction(folder, result, np.array([[1, 1], [4, 9]]), {"fps": 16})
            with np.load(folder / "motion.npz", allow_pickle=False) as motion:
                np.testing.assert_array_equal(motion["trajectories"][0], initial)
                np.testing.assert_array_equal(motion["trajectories"][1:], initial + offsets)
                np.testing.assert_array_equal(motion["pixels_uv"], [[4, 9]])
            with av.open(str(folder / "video.mp4")) as container:
                self.assertEqual(len(list(container.decode(video=0))), 3)
            self.assertEqual(json.loads((folder / "prediction.json").read_text()), {"fps": 16})

    def test_only_ema_weights_survive_conversion(self):
        with tempfile.TemporaryDirectory() as work:
            source, target = Path(work) / "input", Path(work) / "model"
            net = nn.Module()
            net.llm2objectflow = nn.Linear(2, 3, bias=False)
            ema = torch.arange(6, dtype=torch.float32).reshape(3, 2)
            dcp.save(
                {"net.llm2objectflow.weight": torch.zeros_like(ema), "net_ema.llm2objectflow.weight": ema},
                checkpoint_id=source,
                no_dist=True,
            )
            _export_ema(source, target)
            load_weights(net, target)
            torch.testing.assert_close(net.llm2objectflow.weight, ema)
            with self.assertRaisesRegex(ValueError, "Incompatible WorldFlow"):
                load_weights(nn.Linear(2, 3), target)


if __name__ == "__main__":
    unittest.main()
