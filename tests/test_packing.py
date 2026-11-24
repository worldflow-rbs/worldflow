"""Regression for the paper checkpoint's joint rotary coordinates and observation mask."""

import unittest

import torch

from worldflow.models.packing import pack_joint


class JointPackingTests(unittest.TestCase):
    def test_observed_frame_and_pixel_aligned_motion(self):
        # Four visual patches per latent frame. Query the upper-left and
        # lower-right patches of the corresponding 64 × 64 RGB frame.
        packed = pack_joint(
            text_ids=[11, 12],
            vision=torch.zeros(1, 48, 2, 4, 4),
            flow=torch.zeros(4, 2, 3),
            timestep=800,
            points_xyz=torch.tensor([[0.0, 0.0, 1.0], [0.1, 0.1, 1.0]]),
            pixel_indices=torch.tensor([0, 63 * 64 + 63]),
            eos_id=2,
            vision_start_id=3,
            fps=16,
        )
        self.assertEqual(packed.text_ids.tolist(), [11, 12, 2, 3])
        self.assertEqual(packed.split_lens, [4, 16])
        self.assertEqual(packed.attn_modes, ["causal", "full"])

        # SPV1 Nano separates the text and generation clocks by 15000.
        # Video ticks advance by 24/16 per latent; motion advances by 1/4
        # per RGB frame and retains the source query's 32-pixel patch.
        temporal = (
            [0, 1, 2, 3] + [15004] * 4 + [15005.5] * 4 + [15004.25] * 2 + [15004.5] * 2 + [15004.75] * 2 + [15005] * 2
        )
        rows = [0, 1, 2, 3] + [0, 0, 1, 1] * 2 + [0, 1] * 4
        cols = [0, 1, 2, 3] + [0, 1, 0, 1] * 2 + [0, 1] * 4
        expected = torch.tensor([temporal, rows, cols], dtype=torch.float32)
        torch.testing.assert_close(packed.position_ids, expected, atol=0, rtol=0)
        self.assertEqual(packed.vision.condition_mask[0].flatten().tolist(), [1, 0])
        self.assertEqual(packed.objectflow.condition_mask[0].flatten().tolist(), [0, 0, 0, 0])
        self.assertEqual(packed.vision.mse_loss_indexes.tolist(), list(range(8, 12)))
        self.assertEqual(packed.objectflow.mse_loss_indexes.tolist(), list(range(12, 20)))
        self.assertEqual(packed.vision.timesteps.tolist(), [800.0] * 4)
        self.assertEqual(packed.objectflow.timesteps.tolist(), [800.0] * 8)
