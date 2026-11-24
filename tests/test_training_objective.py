"""Regression checks for the stage boundary and joint motion objective."""

import json
import random
import tempfile
import unittest
from pathlib import Path

import torch

from worldflow.inference.loader import checkpoint_stage, validate_bundle, validate_shared_assets
from worldflow.inference.presets import flow_scale, model_prompt
from worldflow.training.augmentation import augment_demo5
from worldflow.training.objective import _training_prompt, _velocity_loss, trajectory_losses


class TrainingObjectiveTests(unittest.TestCase):
    def test_bundle_stage_controls_metric_scale_and_prompt(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory)
            (bundle / "model").mkdir()
            (bundle / "model" / "model.safetensors.index.json").write_text("{}")
            self.assertEqual(checkpoint_stage(bundle), 2)
            manifest = bundle / "worldflow.json"
            manifest.write_text(json.dumps({"stage": 1}))
            self.assertEqual(checkpoint_stage(bundle), 1)
            manifest.write_text(json.dumps({"stage": True}))
            with self.assertRaises(ValueError):
                checkpoint_stage(bundle)
        self.assertEqual(flow_scale("sim7", 1), 0.0155919)
        self.assertEqual(flow_scale("sim7", 2), 0.010821323375216966)
        self.assertEqual(model_prompt("Push the block.", "sim7", stage=1), "Push the block.")
        context = {"origin": "real", "actor": "none", "view": "external_fixed", "scene": "indoor"}
        prompt = model_prompt("Wait.", "demo5", stage=2, prompt_context=context)
        self.assertIn("actor=none", prompt)
        with self.assertRaises(ValueError):
            model_prompt("Wait.", "demo5", prompt_context={"actor": "none"})

    def test_unified_weights_with_only_stage_one(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "tokenizer").mkdir()
            for path in ("tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json", "Wan2.2_VAE.pth"):
                (root / path).touch()
            validate_shared_assets(root)
            first = root / "models/stage1"
            first.mkdir(parents=True)
            (first / "model.safetensors.index.json").write_text("{}")
            validate_bundle(root, 1)
            self.assertEqual(checkpoint_stage(root, 1), 1)
            with self.assertRaises(FileNotFoundError):
                validate_bundle(root)

    def test_dropout_removes_the_whole_structured_prompt(self):
        sample = {"instruction": "Push.", "source": "sim7", "sample_worker_id": 0, "mixed_sample_position": 0}
        self.assertEqual(_training_prompt(sample, 2, 1.0, 42, 0), "")
        self.assertEqual(_training_prompt(sample, 2, 0.0, 42, 0), model_prompt("Push.", "sim7"))

    def test_stage_one_demo5_paraphrase_preserves_motion_and_object_identity(self):
        self.assertEqual(
            augment_demo5("113. 把黑色发夹放在白色纸巾左边", random.Random(0)), "将黑色的发夹移到白色纸巾的左侧"
        )
        self.assertEqual(augment_demo5("Put the cup on the table", random.Random(0)), "Put the cup on the table")

    def test_video_mask_keeps_the_conditioned_frame_in_the_denominator(self):
        pred = torch.tensor([31., 2.], requires_grad=True)
        loss = _velocity_loss(pred, torch.zeros_like(pred), torch.tensor([0., 1.]))
        self.assertEqual(loss.item(), 2.)
        loss.backward()
        torch.testing.assert_close(pred.grad, torch.tensor([0., 2.]))

    def test_trajectory_terms_reduce_weighted_steps_and_ignore_static_endpoints(self):
        # Clip 0 moves +x but is predicted -x. Clip 1 is static and spuriously
        # moves +y. Only clip 0 contributes heading and relative endpoint error.
        truth = torch.zeros(2, 3, 1, 3)
        truth[0, :, 0, 0] = 1
        predicted = -truth.clone()
        predicted[1, :, 0, 1] = 1
        sigma = torch.tensor([0.25, 0.75])
        velocity = ((truth - predicted) / sigma[:, None, None, None]).requires_grad_()
        terms = trajectory_losses(truth, sigma, velocity, torch.zeros_like(velocity), 0.1)
        torch.testing.assert_close(torch.stack(terms), torch.tensor([2., 3.25, 2.]), atol=4e-6, rtol=0.)
        sum(terms).backward()
        self.assertTrue(velocity.grad.isfinite().all())
        self.assertGreater(velocity.grad.abs().sum().item(), 0.)
